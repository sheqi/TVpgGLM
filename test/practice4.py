# Abstract methods/Interface
# overriding

# the childern method is also overidding parent method 
# if not childern method is given, we recall the base method

from abc import ABCMeta, abstractmethod

# Interface
class Shape(object):

	__metaclass__ = ABCMeta

	# this fucntion can be called with all the childern funtion
	@abstractmethod
	def area(self):
		return self.width * self.height

	@abstractmethod
	def perimeter(self):
		return self.width *2 +self.height *2 

class Rectangle(Shape):

	def __init__(self, width, height):

		self.width = width
		self.height = height

		super(Rectangle, self).__init__()

	# def area(self):
	# 	return self.width * self.height

	# def perimeter(self):
	# 	return self.width *2 +self.height *2 

		
class Square(Rectangle):

	def __init__(self, side_length):

		self.side_length = side_length
		super(Square, self).__init__(side_length, side_length)

	# Overide Rectangle area method
	def area(self):
		print("ss")

	 	return(self.side_length)

s = Square(5)

print(s.area())