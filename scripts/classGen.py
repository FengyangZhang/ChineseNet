import os
import sys
import re

def main(argv):
    img_dir = "madeups_gen/"
    img_list = []
    img_names = os.listdir(img_dir)
    dic = {
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        'A': '10',
        'B': '11',
        'C': '12',
        'D': '13',
        'E': '14',
        'F': '15',
        'G': '16',
        'H': '17',
        'J': '18',
        'K': '19',
        'L': '20',
        'M': '21',
        'N': '22',
        'P': '23',
        'Q': '24',
        'R': '25',
        'S': '26',
        'T': '27',
        'U': '28',
        'V': '29',
        'W': '30',
        'X': '31',
        'Y': '32',
        'Z': '33',
        '藏': '34',
        '川': '35',
        '鄂': '36',
        '甘': '37',
        '赣': '38',
        '贵': '39',
        '桂': '40',
        '黑': '41',
        '沪': '42',
        '吉': '43',
        '冀': '44',
        '津': '45',
        '晋': '46',
        '京': '47',
        '辽': '48',
        '鲁': '49',
        '蒙': '50',
        '闽': '51',
        '宁': '52',
        '青': '53',
        '琼': '54',
        '陕': '55',
        '苏': '56',
        '皖': '57',
        '湘': '58',
        '新': '59',
        '渝': '60',
        '豫': '61',
        '粤': '62',
        '云': '63',
        '浙': '64'
    }
    file=open('classes.txt','w')
    is_jpg = re.compile(r'.+?\.jpg')
    print('generating class labels...') 
    if (len(img_names)>0):
        for name in img_names:
            if (is_jpg.match(name)):
                file.write(dic[name[0]])
                file.write('\t')
    print('class label generated.')
if __name__ == "__main__":
    main(sys.argv[1:])