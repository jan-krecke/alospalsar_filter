import numpy as np
import struct


def read_alospalsar_image(file_path):
    fp = open(file_path, mode="rb")

    fp.seek(248)  # ref page 120 of PALSAR_x_Format_EL.pdf
    npixel = int(fp.read(8))
    fp.seek(236)  # ref page 120 of PALSAR_x_Format_EL.pdf
    nrec = (
        412 + npixel * 8
    )  # ref page 122 of PALSAR_x_Format_EL.pdf ... 8 bytes is 32 bits...
    nline = 18432
    fp.seek(720)

    data = struct.unpack(
        ">%s" % (int((nrec * nline) / 4)) + "f", fp.read(int(nrec * nline))
    )  # Read signal data as a 32-bit floating point number

    data = np.array(data).reshape(-1, int(nrec / 4))  # Convert to 2D data
    data = data[:, int(412 / 4) : int(nrec / 4)]  # remove prefix
    slc = data[:, ::2] + 1j * data[:, 1::2]  # separate I and Q components
    return slc
