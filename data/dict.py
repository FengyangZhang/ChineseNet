#!/usr/bin/python
# -*- coding: utf-8 -*-
file = open('chars_in_seq.txt')
a = file.read()
a = unicode(a, 'utf-8')
print(a[1])
