from OpenGL.GL import *
from OpenGL.GLUT import *
from itertools import count
import numpy as np

"""Minimal implementations of OpenGL Texture and Shader classes."""

class Texture:
    idGenerator = count()

    IF_F_T = dict() # internalformat, format, type
    IF_F_T[None] = [GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgba32f"] = [GL_RGBA32F, GL_RGBA, GL_FLOAT]
    IF_F_T["rgb32f"] = [GL_RGB32F, GL_RGB, GL_FLOAT]
    IF_F_T["r32f"] = [GL_R32F, GL_RED, GL_FLOAT]
    IF_F_T["rg32f"] = [GL_RG32F, GL_RG, GL_FLOAT]
    IF_F_T["rgbu16"] = [GL_RGB16UI, GL_RGB_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgba16u"] = [GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT]
    IF_F_T["rgba32u"] = [GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT]
    IF_F_T["r16u"] = [GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_INT]

    def __init__(self, format=None):
        if format:
            self.id = next(Texture.idGenerator)
            self.renderer_id = glGenTextures(1)
            self.width = 0
            self.height = 0
            if_f_t = Texture.IF_F_T[format]
            self.internalformat = if_f_t[0]
            self.format = if_f_t[1]
            self.type = if_f_t[2]

            glBindTexture(GL_TEXTURE_2D, self.renderer_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    def bind(self, slot=0):
        glActiveTexture(GL_TEXTURE0 + int(slot))
        glBindTexture(GL_TEXTURE_2D, self.renderer_id)

    def update(self, pixeldata, width=None, height=None):
        self.bind()
        if width:
            self.width = width
        if height:
            self.height = height
        else:
            self.width = np.shape(pixeldata)[1]
            self.height = np.shape(pixeldata)[0]

        imgdata = None
        if not pixeldata is None:
            imgdata = pixeldata.flatten()

        glTexImage2D(GL_TEXTURE_2D, 0, self.internalformat, self.width, self.height, 0, self.format, self.type, imgdata)


class Shader:
    """Uniforms can only be uploaded when Shaders is first manually bound by user."""
    def __init__(self, sourcecode = None):
        if sourcecode:
            self.shaderProgram = glCreateProgram()
            self.compile(sourcecode)

    def compile(self, sourcecode):
        shaderStages = dict()
        shaderObjects = list()
        currentStage = None
        # Parse source into individual shader stages' source code
        with open(sourcecode, 'r') as source:
            for line in source:
                if "#vertex" in line:
                    currentStage = GL_VERTEX_SHADER
                    shaderStages[currentStage] = ""
                elif "#fragment" in line:
                    currentStage = GL_FRAGMENT_SHADER
                    shaderStages[currentStage] = ""
                elif "#geometry" in line:
                    currentStage = GL_GEOMETRY_SHADER
                    shaderStages[currentStage] = ""
                elif "#compute" in line:
                    currentStage = GL_COMPUTE_SHADER
                    shaderStages[currentStage] = ""
                else:
                    shaderStages[currentStage] += line
        # Compile stages
        for key in shaderStages:
            shaderObjects.append(glCreateShader(key))
            glShaderSource(shaderObjects[-1], shaderStages[key])
            glCompileShader(shaderObjects[-1])
            status = glGetShaderiv(shaderObjects[-1], GL_COMPILE_STATUS)
            if status == GL_FALSE:
                if key == GL_VERTEX_SHADER:
                    strShaderType = "vertex"
                elif key == GL_FRAGMENT_SHADER:
                    strShaderType = "fragment"
                elif key == GL_GEOMETRY_SHADER:
                    strShaderType = "geometry"
                elif key == GL_COMPUTE_SHADER:
                    strShaderType = "compute"
                raise RuntimeError("Shaders compilation failure for type "+strShaderType+":\n" + glGetShaderInfoLog(shaderObjects[-1]).decode('utf-8'))
            glAttachShader(self.shaderProgram, shaderObjects[-1])
        glLinkProgram(self.shaderProgram)
        status = glGetProgramiv(self.shaderProgram, GL_LINK_STATUS)
        if status == GL_FALSE:
            raise RuntimeError("Shaders link failure:\n"+glGetProgramInfoLog(self.shaderProgram).decode('utf-8'))
        for shader in shaderObjects:
            glDetachShader(self.shaderProgram, shader)
            glDeleteShader(shader)

    def bind(self):
        glUseProgram(self.shaderProgram)

    def unbind(self):
        glUseProgram(0)

    def uniform1f(self, uniformName, uniformFloatValue):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform1f(uniformLocation, uniformFloatValue)

    def uniform1i(self, uniformName, uniformIntValue):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform1i(uniformLocation, uniformIntValue)

    def uniform2f(self, uniformName, uniformFloat2Value):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform2f(uniformLocation, uniformFloat2Value[0], uniformFloat2Value[1])

    def uniform3f(self, uniformName, uniformFloat3Value):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniform3f(uniformLocation, uniformFloat3Value[0], uniformFloat3Value[1], uniformFloat3Value[2])

    def uniformmat4(self, uniformName, uniformMat4):
        uniformLocation = glGetUniformLocation(self.shaderProgram, uniformName)
        glUniformMatrix4fv(uniformLocation, 1, GL_TRUE, uniformMat4)

