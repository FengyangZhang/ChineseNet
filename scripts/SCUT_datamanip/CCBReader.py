import sys
import numpy as np
from PIL import Image
import struct
from matplotlib import pyplot as plt

'''
The CCB files are stored in little-endian way,
Where the least significant byte is stored in
the smallest address.
'''
class CCBReader:
	img_size = 4096
	label_length = 2
	address_length = 4
	src = None
	num_chars = 0
	table_offset = 0
	database_name = None
	simp_trad_sign = None


	def __init__(self, src):
		self.src = src
		with open(src, 'rb') as f:
			self.num_chars = struct.unpack('<I', f.read(4))[0]
			self.table_offset = struct.unpack('<I', f.read(4))[0]

	def printInfo(self):
		print("[INFO] Reading from %s, %d characters in total." %(self.src, self.num_chars))
		print("[INFO] The index table is at offset %d" %self.table_offset)

	def getCharImg(self, address):
		with open(self.src, 'rb') as f:
			f.seek(address)
			s = f.read(self.img_size)
			img_array = np.fromstring(s, dtype='<B').reshape((64, 64))
			plt.imshow(img_array, cmap='gray')
			plt.gca().axis('off')
			plt.show()

	def getCharLabelnAddress(self, address):
		with open(self.src, 'rb') as f:
			f.seek(address)
			label = struct.unpack('<H', f.read(self.label_length))[0]
			print("[INFO] The char label(code) is %d" %label)
			address = struct.unpack('<I', f.read(self.address_length))[0]
			print("[INFO] The char address is %d" %address)
			return address

	# def lookUpImgByCode(self, code):

	def lookUpImgByOffset(self, offset):
		if(offset > self.num_chars - 1):
			print("[WARNING] The requested image is out of range!")
		else:
			label_address = self.table_offset + (self.label_length + self.address_length) \
								* offset
			img_address = self.getCharLabelnAddress(label_address)
			print("[INFO] The image is being displayed.")
			self.getCharImg(img_address)


def main(argv):
	reader = CCBReader("./000.ccb")
	reader.printInfo()
	reader.lookUpImgByOffset(5888)

if __name__ == '__main__':
	main(sys.argv[1:])