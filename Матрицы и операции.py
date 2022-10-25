import numpy as np


class Matrix(object):
    '''
    Класс, представляющий собой матрицу и многие методы к ней
    '''

    def __init__(self, matr):
        types = (np.ndarray, list, tuple)

        if type(matr) not in types:
            raise TypeError('Не могу обработать такой тип')

        if type(matr[0]) not in types:
            raise ValueError('Объект не двумерный')

        if type(matr[0][0]) in types:
            raise ValueError('Слишком много измерений')

        a = len(matr[0])
        for row in matr:
            if type(row) not in types:
                raise ValueError('Матрица не целостная')

            if len(row) != a:
                raise ValueError('Объект не прямоугольной формы')
            a = len(row)

            for item in row:
                if type(item) in types:
                    raise ValueError('Где-то лишнее измерение')

        self.matrix = np.array(matr, dtype='float')

    def add_row(self, row):
        # Добавление строки в матрицу
        if len(row) != self.matrix.shape[1]:
            raise TypeError('Длинна строки не совпадает с длинной матрицы!')

        self.matrix = np.vstack([self.matrix, row])

        return Matrix(self.matrix)

    def add_column(self, column):
        # Добавление столбца в матрицу
        if len(column) != self.matrix.shape[0]:
            raise TypeError('Высота столбца не совпадает с высотой матрицы!')

        self.matrix = np.hstack([self.matrix, np.reshape(column, (len(column), 1))])

        return Matrix(self.matrix)

    @property
    def shape(self):
        # Функция для получения формы матрицы
        return self.matrix.shape

    def __matmul__(self, other):
        # Математическое умножение матриц
        if self.matrix.shape[0] != other.matrix.shape[1]:
            raise ValueError('Высота первой матрицы не совпадает с шириной второй!')

        new_M = np.zeros((self.matrix.shape[0], other.matrix.shape[1]))
        for i in range(self.matrix.shape[0]):
            for j in range(other.matrix.shape[1]):
                new_M[i, j] = sum(self.matrix[i, :] * other.matrix[:, j])

        return Matrix(new_M)

    def __mul__(self, other):
        # Умножение матрицы на число или методом Адамара
        if type(other) == Matrix:
            return Matrix(self.matrix * other.matrix)
        else:
            return Matrix(self.matrix * other)

    def __add__(self, other):
        # Сложение матриц
        if self.matrix.shape != other.matrix.shape:
            raise TypeeError('Матрицы разной размерности!')
        new_M = np.zeros(self.matrix.shape)

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                new_M[i, j] = self.matrix[i, j] + other.matrix[i, j]

        return Matrix(new_M)

    def __sub__(self, other):
        # Вычетание матриц
        if self.matrix.shape != other.matrix.shape:
            raise TypeeError('Матрицы разной размерности!')
        new_M = np.zeros(self.matrix.shape)

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                new_M[i, j] = self.matrix[i, j] - other.matrix[i, j]

        return Matrix(new_M)

    @property
    def T(self):
        # Транспонирование матрицы
        TM = np.zeros(self.matrix.shape[::-1])
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                TM[j, i] = self.matrix[i, j]

        return Matrix(TM)

    @property
    def trace(self):
        # След матрицы
        return sum(self.m_diag)

    def kron(self, other):
        # Умножение матриц методом Кронекера
        rows = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(self.matrix[0])):
                row.append(other.matrix * self.matrix[i, j])
            rows.append(row)

        n_rows = []
        for row in rows:
            n_row = row[0]
            for item in row[1:]:
                n_row = np.hstack([n_row, item])
            n_rows.append(n_row)

        new_M = n_rows[0]
        for row in n_rows[1:]:
            new_M = np.vstack([new_M, row])

        return Matrix(new_M)

    @property
    def m_diag(self):
        # Получение главной диагонали матрицы
        tr = []
        for i in range(min(self.matrix.shape)):
            tr.append(self.matrix[i, i])
        return tr

    def __pow__(self, other):
        # Возведение матрицы в степень
        if other < 0:
            raise ValueError('Степень должна быть неотрицательной!')

        if other == 0:
            return Matrix(np.diag(np.ones(min(self.shape))))

        new_M = Matrix(self.matrix)
        for i in range(other - 1):
            new_M = new_M @ Matrix(self.matrix)

        return new_M

    @property
    def inv(self):
        # Обратная матрица
        if self.det == 0:
            raise ValueError('Определитель не должен быть равен нулю!')

        return Matrix(np.linalg.inv(self.matrix))

        # Функция добавлена для полноты функционала, реализована через numpy

    @property
    def det(self):
        # Определитель матрицы
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise TypeError('Матрица должна быть квадратной!')

        return np.linalg.det(self.matrix)

        # Функция добавлена для полноты функционала, реализована через numpy

    def __repr__(self):
        # Функция, отвечающая за вывод данных о матрице
        return str(self.matrix)