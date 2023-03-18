from opengl_classes import *
import glfw
import math
import tifffile
import matplotlib.pyplot as plt
import time

GLFW_CONTEXT_VERSION_MAJOR = 3
GLFW_CONTEXT_VERSION_MINOR = 3
EDGE_MODE = 0  # 0 for zero, 1 for repeat, 2 for reflect
MAXIMUM_ROI = 3
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

def init(create_window=False, image_size=256):
    """
    Initialize pyGPUfit.
    :param create_window: bool. When True, pyGPUfit creates a glfw window context for OpenGL. When using
    :param image_size: int. must be a power of 2. Set the size of the input images. Can be changed after calling init()
    by calling set_image_size(image_size)
    pyGPUfit within a project that already has an OpenGL context, no window needs to be created.
    """
    global window, cs_butterfly, cs_fft, cs_fft_pi, cs_multiply, cs_resample
    if not glfw.init():
        raise Exception("Could not initialize GLFW")

    # create a hidden glfw window and set OpenGL version
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, GLFW_CONTEXT_VERSION_MAJOR)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, GLFW_CONTEXT_VERSION_MINOR)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, OpenGL.GL.GL_TRUE)
    window = glfw.create_window(2, 2, "pyGPUfit hidden window", None, None)
    glfw.make_context_current(window)

    # compile shaders
    shader_dir = os.path.join(os.path.dirname(__file__))+"/shaders"
    cs_butterfly = Shader(shader_dir + "/butterflytexture.glsl")
    cs_fft = Shader(shader_dir + "/fft.glsl")
    cs_fft_pi = Shader(shader_dir + "/fft_inversion_permutation.glsl")
    cs_multiply = Shader(shader_dir + "/fft_phase_correlation.glsl")
    cs_resample = Shader(shader_dir + "/resample_image.glsl")

    # create textures with set_image_size
    set_image_size(image_size) # TODO: remove need for image_size input in init() by automatically re-generating textures if user uses pyGPUfit with a different image size than the active image size

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

def terminate():
    glfw.terminate()


def register(template, subject):
    """
    :param template: image to which the subject image is registered
    :param subject: image that is to be registered onto the template
    :return:
    """


def fft(image, image2=None):
    """
    Compute the fourier transform of the input image(s). Images are uploaded to the GPU as an RGBA32F texture,
    but only two channels are required to store the FFT result for one image (real values and complex values
    are mapped to the R and G colour channels). This means two images can be packed into one texture, so it is
    most efficieny to compute two FFTs at once.
    :param image: numpy array; image to compute the fourier transform of. Values should be np.float32 type
    :param image2: numpy array; optional second image to also compute the fourier transform of. Values should be np.float32 type
    :return: TODO
    """
    if image.shape != (image_size, image_size):
        raise Exception(f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUfit.set_image_size() to change the expected image size")
    if image2 is not None and image2.shape != (image_size, image_size):
        raise Exception(f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUfit.set_image_size() to change the expected image size")

    data = np.zeros((image_size, image_size, 4))
    data[:, :, 0] = image
    if image2 is not None:
        data[:, :, 2] = image2

    texture_data.update(data)

    # launch the FFT shader and compute horizontal FFTs
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
    cs_fft_pi.bind()

    cs_fft_pi.uniform1i("pingpong", pingpong % 2)
    cs_fft_pi.uniform1f("N", float(image_size))
    glBindImageTexture(0, texture_data.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
    glBindImageTexture(1, texture_data.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glBindImageTexture(2, texture_data_buffer.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    texture_data.bind()
    FTs = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    return FTs[:, :, 0:2], FTs[:, :, 2:]


def read_and_save_texture_to_temp_folder(tex, title):
    tex.bind()
    a = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    tifffile.imwrite(f"C:/Users/mgflast/PycharmProjects/pyGPUreg/temp/{title}_r.tif", a[:, :])

def cross_correlation(image, image2):
    """
    Uses the exact same code as fft(), but instead of returning the FTs it also computes the product of the two FTs,
    and then computes and returns the inverse FT; i.e., this function returns the cross correlation of image and image2
    """
    def compute_ffts(do_inversion_permutation=True):
        """
        Helper function - since we compute the forward and reverse FFTs in the cross_correlation function, and these
        use almost exactly the same code, we wrap in into compute_fft(). Note that we don't _actually_ compute the
        reverse FFT, as we don't do any amplitude scaling. In the end we're only interested in finding the maximum
        intensity value in the final cross correlation image, and amplitude scaling doesn't affect that.
        Before calling compute_fft, we manually set up the textures on the GPU side - see usage of this function
        in the below code for the parent, cross_correlation(), function.
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

    if image.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUfit.set_image_size() to change the expected image size")
    if image2 is not None and image2.shape != (image_size, image_size):
        raise Exception(
            f"Image size is {image.shape} but should be ({image_size}, {image_size}). Call pyGPUfit.set_image_size() to change the expected image size")

    data = np.zeros((image_size, image_size, 4))
    data[:, :, 0] = image
    if image2 is not None:
        data[:, :, 2] = image2

    texture_data.update(data)

    # forward FFTs
    compute_ffts()

    # use the cs_multiply compute shader to calculate the product of the Fourier transforms
    cs_multiply.bind()
    glBindImageTexture(0, texture_data.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    cs_multiply.unbind()
    # forward FFTs again to get cross correlation (without amplitude scaling.)
    compute_ffts(do_inversion_permutation=False)

    # get amplitude correlation image and find maximum
    xcorr = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    def subpixel_maximum(img):
        pass # TODO

    # get the sub-pixel location of the peak
    shift = np.array(np.unravel_index(np.argmax(xcorr), xcorr.shape))

    positive_shift = np.array(shift)
    negative_shift = positive_shift - image_size
    mag_pos = np.sum(positive_shift ** 2)
    mag_neg = np.sum(negative_shift ** 2)
    if mag_pos < mag_neg:
        shift = positive_shift
    else:
        shift = negative_shift
    print(shift)
    # re-sample the image with a shift
    texture_resample_a.update(image2)
    texture_resample_b.update(None, width=image_size, height=image_size)
    cs_resample.bind()
    cs_resample.uniform1i("dx", shift[0])
    cs_resample.uniform1i("dy", shift[1])
    cs_resample.uniform1i("N", image_size)
    cs_resample.uniform1i("edge_mode", EDGE_MODE)
    glBindImageTexture(0, texture_resample_a.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F)
    glBindImageTexture(1, texture_resample_b.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F)
    glDispatchCompute(*compute_space_size)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    cs_resample.unbind()

    texture_resample_b.bind()
    resampled = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)

    return resampled



