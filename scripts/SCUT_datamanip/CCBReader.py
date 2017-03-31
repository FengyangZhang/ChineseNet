import sys
import numpy as np
from PIL import Image
import struct
from matplotlib import pyplot as plt
import binascii

'''
The CCB files are stored in little-endian way,
Where the least significant byte is stored in
the smallest address.
'''
class CCBReader:
	img_height = 64
	img_width = 64
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
			self.database_name = struct.unpack('<50s', f.read(50))[0].decode('gbk')
			self.simp_trad_sign = struct.unpack('<10s', f.read(10))[0].decode('gbk')

	def printBasicInfo(self):
		print("[INFO] Reading from %s, %d characters in total." %(self.src, self.num_chars))
		# print("[INFO] The index table is at offset %d" %self.table_offset)
		print("[INFO] The database name is %s" %self.database_name)
		print("[INFO] The database contains characters in format: %s" %self.simp_trad_sign)

	def getCharImg(self, address):
		with open(self.src, 'rb') as f:
			f.seek(address)
			s = f.read(self.img_size)
			img_array = np.fromstring(s, dtype='<B').reshape((self.img_height, self.img_width))
			# plt.imshow(img_array, cmap='gray')
			# plt.gca().axis('off')
			# plt.show()
			return img_array

	def getCharLabelnAddress(self, address):
		with open(self.src, 'rb') as f:
			f.seek(address)
			label = struct.unpack('H', f.read(self.label_length))
			print("[INFO] The char label(code) is %s(in little-endian)" %label)
			img_address = struct.unpack('<I', f.read(self.address_length))[0]
			# print("[INFO] The char address is %d" %img_address)
			return img_address, label

	# def lookUpImgByCode(self, code):

	def lookUpImgByOffset(self, offset):
		if(offset > self.num_chars - 1):
			print("[WARNING] The requested image is out of range!")
		else:
			label_address = self.table_offset + (self.label_length + self.address_length) \
								* offset
			img_address, _ = self.getCharLabelnAddress(label_address)
			print("[INFO] The image is being displayed.")
			return self.getCharImg(img_address)