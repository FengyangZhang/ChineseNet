import sys
from CCBReader import CCBReader
from dataGenerator import dataGenerator

def main(argv):
	reader = CCBReader("./000.ccb")
	generator = dataGenerator('./', './')
	# reader.printBasicInfo()
	# reader.lookUpImgByOffset(0)

if __name__ == '__main__':
	main(sys.argv[1:])