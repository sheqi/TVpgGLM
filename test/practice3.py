
# object 
# car

# superclass car

class Car(object):
	
	def __init__(self, car_type, color):

		self.car_type = car_type
		self.color = color

	def drive(self):
		print("Driving my %s %s" % (self.color, self.car_type))

	def park(self):
		print("Parked car")

# Subcalss of car called Honda

class Honda(Car):

	# this only initialize for this input of class 
	def __init__(self, color):

		super(Honda, self).__init__("Honda", color)


mycar = Honda("Yellow")

mycar.drive()
mycar.park()