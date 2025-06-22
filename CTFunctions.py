import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from scipy.sparse import csr_matrix


class lines:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.steigung = self.__berechne_steigung()
        self.t = self.__berechne_t()

    def __berechne_steigung(self):
        try:
            if (self.end_pos[0] - self.start_pos[0]) == 0:
                return None
            return (self.end_pos[1] - self.start_pos[1]) / (self.end_pos[0] - self.start_pos[0])
        except:
            return None

    def __berechne_t(self):
        if self.steigung == None:
            return None
        return self.start_pos[1] - (self.steigung * self.start_pos[0])

    def berechne_punkt_linie(self, x):
        if self.steigung == None:
            return None
        punkt = (self.steigung * x) + self.t
        return punkt

    def __str__(self):
        return f"gerade Gleichung ist: {self.steigung}*x" + " +" + f"{self.t}"


def define_circle(img):
    width, height = img.shape[0], img.shape[1]
    zentrum = np.array([int(width/2), int(width/2)])
    radius = np.sqrt(((width - zentrum[0])**2) + ((height - zentrum[1])**2))
    thetha = np.linspace(0, 2*np.pi, 100)
    x = zentrum[0] + radius*np.cos(thetha)
    y = zentrum[1] + radius*np.sin(thetha)

    return x, y, zentrum, radius

def get_start_lines(img, amount_lines):
    x, y, zentrum, radius = define_circle(img)
    x_values = np.linspace(zentrum[0] - radius, zentrum[0] + radius, amount_lines)
    start_lines = []
    for x in x_values:
    # y-Werte berechnen, wo die Gerade den Kreis schneidet:
        y_offset = np.sqrt(radius**2 - (x - zentrum[0])**2)
        y1 = zentrum[1] - y_offset
        y2 = zentrum[1] + y_offset
        start_lines.append([[x, x], [y1, y2]])
    return start_lines

def rotate_lines(img, lines, winkel, zentrum):
    width, height = img.shape
    test = np.array(lines)
    rotate_matrix = np.array([[np.cos(np.radians(winkel)), -np.sin(np.radians(winkel))], [np.sin(np.radians(winkel)), np.cos(np.radians(winkel))]])

    lines = lines - zentrum

    rotation_lines = rotate_matrix@lines

    rotation_lines = rotation_lines + zentrum

    return rotation_lines


def kalkuliere_Matrix_s(img, x_wert, y_wert, anzahl_start_lines, amount_degrees):
    width, height = img.shape
    dimension = ((amount_degrees * anzahl_start_lines), (width * height))
    Matrix_s = np.zeros(dimension)
    for kk in range(len(x_wert)):
        if len(x_wert[kk]) == 0:
            continue
        current_pixel = [x_wert[kk][0], y_wert[kk][0]]
        for ll in range(1, len(x_wert[kk])):
            pixel = [int(x_wert[kk][ll]), int(y_wert[kk][ll])]
            if [int(current_pixel[0]), int(current_pixel[1])] == pixel:
                continue
            else:
                laenge_line = norm(np.array(current_pixel) - np.array([x_wert[kk][ll - 1], y_wert[kk][ll - 1]]))
                current_pixel = [x_wert[kk][ll], y_wert[kk][ll]]
                if pixel[0] == 0:
                    position = pixel[1]
                else:
                    position = ((pixel[0] - 1) * width) + pixel[1]

            Matrix_s[kk][position] = laenge_line

    return Matrix_s

def kalkuliere_b_vektor(img, Matrix, anzahl_start_lines, amount_degrees):
    width, height = img.shape
    img_reshaped = img.reshape((width*height), 1)
    b = np.zeros(((anzahl_start_lines*amount_degrees), 1))
    S_in = np.sqrt(((width**2) + (height**2)))
    for i in range(Matrix.shape[0]):
        abschwaechung = Matrix[i]@img_reshaped
        if abschwaechung > S_in:
            raise ValueError("Value must be larger than 0")
        S_out = S_in - abschwaechung
        b[i] = np.log((S_in/S_out))
    return b


def berechne_GS(img, anzahl_start_lines, amount_degrees):
    width, height = img.shape
    start_lines = get_start_lines(img, anzahl_start_lines)
    x, y, zentrum, radius = define_circle(img)
    liste = []
    dictionary = {}
    save_all_xwert = []
    save_all_ywert = []
    dimension = ((amount_degrees * anzahl_start_lines), (width * height))
    Matrix_s = np.zeros(dimension)
    degrees = np.linspace(0, 90, amount_degrees)
    for i in degrees:
        dictionary[i] = []
        rotation_lines = rotate_lines(img, start_lines, i, zentrum)
        for k in range(1, len(rotation_lines)):
            line = lines([rotation_lines[k, 0][0], rotation_lines[k, 1][0]],
                         [rotation_lines[k, 0][1], rotation_lines[k, 1][1]])
            if line.steigung == None:
                break
            laenge_line_complete = norm(np.array(line.end_pos) - np.array(line.start_pos))
            log = np.log10(laenge_line_complete)
            schrittweite = laenge_line_complete / ((10) ** (log + 3))
            m = 0
            x_wert_list = []
            y_wert_list = []
            if line.steigung < 0:
                while (rotation_lines[k, 0][1] + m) <= rotation_lines[k, 0][0]:
                    m += schrittweite
                    x_wert = rotation_lines[k, 0][1] + m
                    if (x_wert >= 0) and (x_wert <= width):
                        y_wert = line.berechne_punkt_linie(x_wert)
                        if (y_wert >= 0) and (y_wert <= height):
                            x_wert_list.append(x_wert)
                            y_wert_list.append(y_wert)

            save_all_xwert.append(x_wert_list)
            save_all_ywert.append(y_wert_list)

    Matrix_s = csr_matrix(
        kalkuliere_Matrix_s(img, save_all_xwert, save_all_ywert, anzahl_start_lines, amount_degrees))

    b = kalkuliere_b_vektor(img, Matrix_s, anzahl_start_lines, amount_degrees)

    return Matrix_s, b

