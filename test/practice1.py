# Practice 1

x = 56
def change_value(num):
	num = 90

print(x)
change_value(x)
print(x)

class Boat():
	def __init__(self):
		self.cargo_weight = 23

def change_cargo_weight(ship):
	ship.cargo_weight = 45

boat = Boat()
print(boat.cargo_weight)
change_cargo_weight(boat)
print(boat.cargo_weight)

# Prarice 2

class Human():

	def __init__(self, name, gender):
		
		self.name = name
		self.gender = gender

	def speak_name(self):
		print("My name is %s" % self.name)

	def speak(self, text):
		print(text)
	# method
	def perform_math_task(self, math_operation, *args):
		print("%s performed math and the results were %f" % (self.name, math_operation(*args)))

# function
def add(a, b):
	return a + b

ryan = Human("Ryan steven", "Male")

ryan.perform_math_task(add, 1, 3)

# Practice 3

class Rectangle():

	# Implicit arguments
	def __init__(self, width, length):

		self.width = width
		self.length = length

	def area(self):
		return self.width * self.length

	def perimeter(self):
		return self.length * 2 + self.width * 2

my_rect = Rectangle(5, 6) 
another_rect = Rectangle(2, 10)

print("The area is %f" % (my_rect.area())); print(another_rect.area())
print(my_rect.perimeter()); print(another_rect.perimeter())

# class # car class - # wheels
# instance # House class - house address
# instance # House class - purchase price
# class # Character class - Max health
# instance # Grid class - width and height
# instance # Polygon class - # vertices
# class # Triangle class - ["right triangle", "Scaleene" ...]
# class # Math class - Goledn ratio const

class Character ():

	# static/class variables
	total_number_of_characters = 0
	MAX_HEALTH = 150

	def __init__(self, name):

		# instance variables
		self.name = name
		self.healh = Character.MAX_HEALTH

		Character.total_number_of_characters += 1 

bob = Character("Bob")



















		


