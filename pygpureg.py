from opengl_classes import *
import glfw
import math
import matplotlib.pyplot as plt
import time
import tifffile
# TODO: low pass filter before reg

# Module enums
EDGE_MODE_ZERO = 0
EDGE_MODE_CLAMP = 1
EDGE_MODE_REFLECT = 2
SUBPIXEL_MODE_NONE = 0  # no subpixel peak detection. limits shift resolution to pixel size.
SUBPIXEL_MODE_COM = 1  # detect sub-pixel shift by calculating the center of mass of the amplitude peak
INTERPOLATION_MODE_NEAREST = 0
INTERPOLATION_MODE_LINEAR = 1

# Advanced settings
COM_RADIUS = 2
GLFW_CONTEXT_VERSION_MAJOR = 3  # user can change the GLFW/GL version prior to calling pyGPUreg.init() but code was not tested with any other version than 3.3
GLFW_CONTEXT_VERSION_MINOR = 3  # user can change the GLFW/GL version prior to calling pyGPUreg.init() but code was not tested with any other version than 3.3

# globals: texture, shaders, some misc. vars
window = None
image_size = 0
log_image_size = 0
compute_space_size = (16, 16, 1)
cs_butterfly: Shader
cs_fft: Shader
cs_fft_pi: Shader
cs_multiply: Shader
cs_resample: Shader
texture_butterfly: Texture
texture_data: Texture
texture_data_buffer: Texture
texture_resample_a: Texture
texture_resample_b: Texture


def init(create_window=True, image_size=None):
    """
    Initialize pyGPUreg.
    :param create_window: bool (default True). When True, pyGPUreg creates a glfw window context for OpenGL. When using pyGPUreg within a project that already has an OpenGL context, no window needs to be created.
    :param image_size: None or int. must be a power of 2. Specify the size of the input images in order to reserve space on the GPU. Can be changed after calling init() by calling set_image_size(image_size) - is also changed automatically when cross_correlation() is called with arguments of a different shape than the previously set image size. When None, textures are not initialized until they are actually needed.
    """
    global window, cs_butterfly, cs_fft, cs_fft_pi, cs_multiply, cs_resample
    if create_window:
        if not glfw.init():
            raise Exception("Could not initialize GLFW")

        # create a hidden glfw window and set OpenGL version
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, GLFW_CONTEXT_VERSION_MAJOR)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, GLFW_CONTEXT_VERSION_MINOR)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        window = glfw.create_window(2, 2, "pyGPUreg hidden window", None, None)
        glfw.make_context_current(window)

    # compile shaders
    shader_dir = os.path.join(os.path.dirname(__file__))+"/shaders"
    cs_butterfly = Shader(shader_dir + "/butterflytexture.glsl")
    cs_fft = Shader(shader_dir + "/fft.glsl")
    cs_fft_pi = Shader(shader_dir + "/fft_inversion_permutation.glsl")
    cs_multiply = Shader(shader_dir + "/fft_phase_correlation.glsl")
    cs_resample = Shader(shader_dir + "/resample_image.glsl")

    # create textures with set_image_size
    if image_size is not None:
        set_image_size(image_size)


def set_image_size(size):
    global image_size, log_image_size, compute_space_size, texture_butterfly, texture_data, texture_data_buffer, texture_resample_a, texture_resample_b

    # check size
    N = int(size)
    logN = int(math.log2(N))
    if 2**logN != N:
        raise Exception("Image size must be a power of two.")
    image_size = N
    log_image_size = logN
    compute_space_size = (N // 16, N // 16, 1)

    # create textures
    texture_butterfly = Texture(format="rgba32f")
    texture_butterfly.update(pixeldata=None, width=logN, height=N)
    texture_data = Texture(format="rgba32f")
    texture_data.update(pixeldata=None, width=N, height=N)
    texture_data_buffer = Texture(format="rgba32f")
    texture_data_buffer.update(pixeldata=None, width=N, height=N)
    texture_resample_a = Texture(format="r32f")
    texture_resample_a.update(pixeldata=None, width=N, height=N)
    texture_resample_b = Texture(format="r32f")
    texture_resample_b.update(pixeldata=None, width=N, height=N)

    # fill butterfly texture
    def bit_reverse(val, N):
        n = int(math.log2(N))
        reverse = 0
        for i in range(n):
            if (val % 2) == 1:
                reverse += 2 ** (n - i - 1)
            val //= 2
        return reverse

    cs_butterfly.bind()
    cs_butterfly.uniform1i("N", N)

    bit_reversed_indices = list()
    for n in range(N):
        bit_reversed_indices.append(bit_reverse(n, N))
    bit_reversed_indices = np.asarray(bit_reversed_indices, dtype=np.uint32)

    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, bit_reversed_indices, GL_STREAM_READ)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo)
    glBindImageTexture(0, texture_butterfly.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    cs_butterfly.unbind()


def sample_image_with_shift(image, shift, edge_mode=EDGE_MODE_ZERO, interpolation_mode=INTERPOLATION_MODE_LINEAR):
    """
    :param image: 2D numpy array of pixel values
    :param shift: tuple (dx, dy), the image shift in pixel units
    :param edge_mode: one of: pypgureg.EDGE_MODE_ZERO, .EDGE_MODE_CLAMP, or .EDGE_MODE_REPEAT.
    :param interpolation_mode: one of: pypgureg.INTERPOLATION_MODE_LINEAR (default), .INTERPOLATION_MODE_NEAREST. Note that combining nearest neighbour interpolation with a shift less than one pixel causes the output to be identical to the input image.
    :return: numpy array of pixel values in the resampled image.
    """
    # next up: re-sampling the image and apply the detected shift.
    texture_resample_a.update(image)  # upload the image to be resampled to the gpu
    texture_resample_a.bind()
    # set edge and interpolation mode for that texture
    if edge_mode in [EDGE_MODE_ZERO, EDGE_MODE_CLAMP]:
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    elif edge_mode == EDGE_MODE_REFLECT:
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
    if interpolation_mode == INTERPOLATION_MODE_NEAREST:
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    elif interpolation_mode == INTERPOLATION_MODE_LINEAR:
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    texture_resample_b.update(None, width=image_size, height=image_size) # empty the buffer texture into which the original image is copied
    cs_resample.bind()  # bind compute shader, upload uniforms, bind textures, dispatch
    cs_resample.uniform1f("dx", float(shift[0]))
    cs_resample.uniform1f("dy", float(shift[1]))
    cs_resample.uniform1i("N", image_size)
    cs_resample.uniform1i("edge_mode", edge_mode)
    texture_resample_a.bind(0)
    glBindImageTexture(1, texture_resample_b.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    cs_resample.unbind()

    # copy result to CPU
    texture_resample_b.bind()
    resampled = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)

    return resampled


def bind_and_launch_fft_compute_shaders(do_inversion_permutation=True):
    """
    Helper function - since we compute both forward and reverse FFTs in the cross_correlation function, and these
    use almost exactly the same code, we wrap in into compute_fft(). Note that we don't _really_ compute the
    reverse FFT, as we don't do any amplitude scaling. In the end we're only interested in finding the maximum
    intensity value in the final cross correlation image, and amplitude scaling doesn't affect that.
    Before calling this function we manually set up the textures on the GPU side.
    See usage of this function in the cross_correlation() or the gpu_fft() functions
    :param do_inversion_permutation: bool (default True). Set to False to launch the FFT compute shader, but not the
    inversion and permutation compute shader.
    """
    cs_fft.bind()
    cs_fft.uniform1i("direction", 0)
    glBindImageTexture(0, texture_butterfly.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glBindImageTexture(1, texture_data.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
    glBindImageTexture(2, texture_data_buffer.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
    pingpong = 0
    for s in range(log_image_size):
        cs_fft.uniform1i("pingpong", pingpong % 2)
        cs_fft.uniform1i("stage", s)
        glDispatchCompute(*compute_space_size)
        pingpong += 1

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # next, compute vertical FFTs
    cs_fft.uniform1i("direction", 1)
    for s in range(log_image_size):
        cs_fft.uniform1i("pingpong", pingpong % 2)
        cs_fft.uniform1i("stage", s)
        glDispatchCompute(*compute_space_size)
        pingpong += 1

    cs_fft.unbind()

    if not do_inversion_permutation:
        return

    # do inversion and permutation
    cs_fft_pi.bind()
    cs_fft_pi.uniform1i("pingpong", pingpong % 2)
    cs_fft_pi.uniform1f("N", float(image_size))
    glBindImageTexture(0, texture_data.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
    glBindImageTexture(1, texture_data.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glBindImageTexture(2, texture_data_buffer.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    cs_fft_pi.unbind()


def gpu_fft(image, image2=None):
    """
    Compute a FFT on the GPU for one of two input images. The data is sent to the GPU in the form of an RGBA, float32
    texture. Since one FFT requires only two colour channels (one for the real and one for the imaginary part), it is
    efficient to wrap two images into one texture and compute the FFTs in batches of two.
    :param image: 2D numpy array. Must be square and with size a power of 2.
    :param image2: optional second image
    :return: numpy array with dimensions (image_size, image_size, 2), where index 0 on axis 2 represents the real valued
    part of the FT and index 1 on axis 2 represents the imaginary value. When processing two images in one go, returns a
    tuple of two such numpy arrays instead: (FFT1, FFT2).
    """
    if image_size != image.shape[0]:
        set_image_size(image.shape[0])

    if image.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUreg.set_image_size() to change the expected image size")
    if image2 is not None and image2.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUreg.set_image_size() to change the expected image size")

    data = np.zeros((image_size, image_size, 4))
    data[:, :, 0] = image
    if image2 is not None:
        data[:, :, 2] = image2

    texture_data.update(data)

    bind_and_launch_fft_compute_shaders()

    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT)

    texture_data.bind()
    fourier_transforms = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    if image2 is not None:
        return fourier_transforms[:, :, 0] + 1j * fourier_transforms[:, :, 1], fourier_transforms[:, :, 2] + 1j * fourier_transforms[:, :, 3]
    else:
        return fourier_transforms[:, :, 0] + 1j * fourier_transforms[:, :, 1]


def phase_correlation(template_image, moved_image, apply_shift=True, edge_mode=EDGE_MODE_ZERO, subpixel_mode=SUBPIXEL_MODE_COM, interpolation_mode=INTERPOLATION_MODE_LINEAR):
    """
    Compute the phase correlation of a template and moved image in order to detect the shift between template and moved image. Optionally also resamples the moved image to undo the shift.
    :param template_image: 2d numpy array of pixel data for the template image.
    :param moved_image: 2d numpy array of pixel data for the image that is to be moved onto the template image.
    :param apply_shift: bool (default True), when True, the function returns a re-sampled version of image2 with the detected shift applied. When False, function returns the shift (dx, dy)
    :param edge_mode: one of pyGPUreg.EDGE_MODE_ZERO (default), .EDGE_MODE_CLAMP, or .EDGE_MODE_REFLECT. Zero: pixels outside of original image are set to zero. Clamp: pixel values are clamped to the edge of the original image. Reflect: image is reflected along the edges.
    :param subpixel_mode: one of pyGPUreg.SUBPIXEL_MODE_NONE, .SUBPIXEL_MODE_COM (default). None: amplitude shift is detected with pixel accuracy, meaning the shift resolution is at best 1 pixel. COM: Center of mass - peak is detected and the sub-pixel position estimated based on the center of mass of the peak. Edit the value of pyGPUreg.COM_RADIUS (int, default 2) to change the radius of the mask used in the c.o.m. estimation.
    :param interpolation_mode: either pygpureg.INTERPOLATION_MODE_LINEAR (default) or pygpureg.INTERPOLATION_MODE_NEAREST. Note that when a detected shift is less than a pixel, resampling the image with Nearest interpolation will have no effect.
    :return: shift (dx, dy) if apply_shift=False, else tuple (registered_image, (dx, dy))) with registered_image as a numpy array
    """
    if image_size != template_image.shape[0]:
        set_image_size(template_image.shape[0])


    if template_image.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {template_image.shape} but should be ({image_size}, {image_size}). Call pyGPUreg.set_image_size() to change the expected image size")
    if moved_image.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {template_image.shape} but should be ({image_size}, {image_size}). Call pyGPUreg.set_image_size() to change the expected image size")

    data = np.zeros((image_size, image_size, 4))
    data[:, :, 0] = template_image
    if moved_image is not None:
        data[:, :, 2] = moved_image

    texture_data.update(data)

    # forward FFTs
    bind_and_launch_fft_compute_shaders()

    # use the cs_multiply compute shader to calculate the product of the Fourier transforms
    cs_multiply.bind()
    glBindImageTexture(0, texture_data.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    cs_multiply.unbind()

    # forward FFTs again to get cross correlation (without amplitude scaling.)
    bind_and_launch_fft_compute_shaders(do_inversion_permutation=False)

    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT)

    def detect_subpixel_maximum(phase_correlation, mode):
        shift = np.zeros(2)
        if mode == SUBPIXEL_MODE_NONE:
            shift = np.array(np.unravel_index(np.argmax(phase_correlation), phase_correlation.shape))  # find index of maximum in xcorr, convert to xy coordinate, convert to np array
        elif mode == SUBPIXEL_MODE_COM:
            mass = 0
            peak_xy = np.array(np.unravel_index(np.argmax(phase_correlation), phase_correlation.shape))
            x_coords = np.mod(np.array(range(max([0, peak_xy[0] - COM_RADIUS]), min([image_size, peak_xy[0] + COM_RADIUS + 1]))), image_size)
            y_coords = np.mod(np.array(range(max([0, peak_xy[1] - COM_RADIUS]), min([image_size, peak_xy[1] + COM_RADIUS + 1]))), image_size)
            for x in x_coords:
                for y in y_coords:
                    pc_val = phase_correlation[x, y]
                    shift[0] += x * pc_val
                    shift[1] += y * pc_val
                    mass += pc_val
            shift /= mass

        x_shift = image_size / 2 - shift[0]
        y_shift = image_size / 2 - shift[1]
        if log_image_size % 2 == 0:  # when logN is even, change the sign of the shift.
            x_shift = -x_shift
            y_shift = -y_shift
        return x_shift, y_shift

    # get amplitude correlation image and find maximum
    texture_data.bind()
    xcorr = np.fft.fftshift(glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT), axes=(0, 1))
    xcorr[image_size//2, image_size//2] = 0  ## todo: when template and image are the same, xcorr has (almost) all intensity at (iamge_size//2, image_size//2) - setting that value to 0 leads to peak being in a random location; i.e., registering to self returns nonsense
    dx, dy = detect_subpixel_maximum(xcorr, mode=subpixel_mode)
    if not apply_shift:
        return dx, dy
    resampled = sample_image_with_shift(moved_image, (dx, dy), edge_mode, interpolation_mode)

    return resampled, (dx, dy)



