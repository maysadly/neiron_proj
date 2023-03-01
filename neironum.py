import os
import numpy as np
import pygame
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten


#выбор режима логгирования фреймворка
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#класс клеточного холста для рисования
class Board:
	#Конструктор
	def __init__(self, width, height):
		self.width = width
		self.height = height
		#создание многомерного массива где значения это градация серого
		self.board = [[255] * width for _ in range(height)]
		self.left = 10
		self.top = 10
		self.cell_size = 30
		self.tool = True
		self.rate = 0
		self.res = -1
	
	#Настройка холста
	def set_view(self, left, top, cell_size):
		self.left = left
		self.top = top
		self.cell_size = cell_size
	
	#Показ холста
	def render(self, screen):
		serf = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
		#Входные данные для нейронной сети
		x_arr = []
		for i in range(self.height):
			st = []
			for j in range(self.width):
				gr = self.board[i][j]
				rect = pygame.Rect((j * self.cell_size), (i * self.cell_size), self.cell_size, self.cell_size)
				pygame.draw.rect(serf, pygame.Color(gr, gr, gr), rect, width=0)
				st.append((255 - gr) / 255)
			x_arr.append(st)
		#Частота входных данных
		if self.rate != 100:
			self.rate += 1
		else:
			#Загрузка входных данных в модель нейронной сети
			x_arr = np.array(x_arr)
			self.rate = 0
			x = np.expand_dims(x_arr, axis=0)
			#Результат
			self.res = model.predict(x)
			#Нахождение индекста максимального значения в выходных данных которая является ответом
			self.res = np.argmax(self.res)
		#Вывод ответа
		font = pygame.font.Font(None, 40)
		text = font.render(f"Вывод: {self.res}", True, (100, 255, 100))
		screen.blit(serf, (self.left, self.top))
		screen.blit(text, (70, 590))
	
	#Поменять прибор рисования
	def swap_tool(self, t):
		self.tool = t

	def get_cell(self, mouse_pos):
		x, y = mouse_pos[0] - self.left, mouse_pos[1] - self.top
		xp = 0 <= x < self.width * self.cell_size
		yp = 0 <= y < self.height * self.cell_size
		if xp and yp:
			return (x // self.cell_size, y // self.cell_size)
		else:
			return None
		
	#Проверка на выход из границы
	def new_cell(self, cell):
		if 0 <= cell[0] < self.width and 0 <= cell[1] < self.height:
			return True
		return False

	#Кисть для рисования с жирностью
	def on_click(self, cell):
		if cell is not None:
				for i in (-1, 0, 1):
					for j in (-1, 0, 1):
						if self.new_cell((cell[1] + i, cell[0] + j)):
							hg = i == 0 or j == 0
							if not self.tool:
								if hg:
									self.board[cell[1] + i][cell[0] + j] += 17
								else:
									self.board[cell[1] + i][cell[0] + j] += 5
								if self.board[cell[1] + i][cell[0] + j] > 255:
									self.board[cell[1] + i][cell[0] + j] = 255
							else:
								if hg:
									self.board[cell[1] + i][cell[0] + j] -= 17
								else:
									self.board[cell[1] + i][cell[0] + j] -= 5
								if self.board[cell[1] + i][cell[0] + j] < 0:
									self.board[cell[1] + i][cell[0] + j] = 0

	def get_click(self, mouse_pos):
		cell = self.get_cell(mouse_pos)
		self.on_click(cell)


#Основная программа
if __name__ == '__main__':
	#Загрузка дата сетов для обучения (Матрица 28 на 28 с значениями градации серого от 0 до 255) с рукописными цифрами
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#Нормализация значении в матрице для диапозона значении от 0 до 1
	x_train = x_train / 255
	x_test = x_test / 255
	#Выбор типа выходных данных в нашем случай массив из нулей с правильным ответом единицей в соответсующем индексе
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)
	#Постройка модели нейронной сети
	model = keras.Sequential([
		Flatten(input_shape=(28, 28, 1)),
		Dense(128, activation='relu'),
		Dense(234, activation='relu'),
		Dense(10, activation='softmax')
	])
	#Логирование и просмотр количеств параметров
	print(model.summary())
	#Обучение нейронной сети и настройка обучения
	model.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
	#Обучение с 3 эпохами и с корректировкой весов каждые 32 задачи
	model.fit(x_train, y_train_cat, batch_size=32, epochs=3, validation_split=0.2)
	model.evaluate(x_test, y_test_cat)
	#Инициализация пайгейм и работа с ней
	pygame.init()
	pygame.display.set_caption('Распознавание рукописных цифр')
	screen = pygame.display.set_mode((560, 660))
	board = Board(28, 28)
	board.set_view(0, 0, 20)
	running = True
	#Переменная для фиксирования зажатия кнопки мыши
	sk = (False, -1)
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					sk = (True, 1)
				elif event.button == 3:
					sk = (True, 3)
			if event.type == pygame.MOUSEBUTTONUP:
					sk = (False, -1)
		#Выбор инструмента стерка кисть
		if sk[0]:
			if sk[1] == 1:
				board.swap_tool(True)
				board.get_click(pygame.mouse.get_pos())
			else:
				board.swap_tool(False)
				board.get_click(pygame.mouse.get_pos())
		screen.fill((0, 0, 0))
		board.render(screen)
		pygame.display.flip()
	pygame.quit()