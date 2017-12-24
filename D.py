#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
   Decode関連
'''

#汎用
import sys
import re
import time as t
import base64
from string import ascii_lowercase as lc, ascii_uppercase as uc, ascii_letters as luc, digits as dig
from math import ceil, sqrt
from pprint import pprint

#自作
import Dict as di

#利用しそうなので定義しておく
b64char=uc+lc+dig+'+'+'-'

#-----------------------------------------
# パスコードフォーマット定義
# print(m.group('basename')) #(?P<basename> )にマッチした文字列
# re.Xは、柔軟な解釈。re.Iは大・小文字無視。
fmt_normal =re.compile(r"""
  ^
  (?P<num0>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<pre0>[a-z]{3})
  (?P<num1>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<keyword>.+) # キーワード
  (?P<pos0>[a-z])
  (?P<num2>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<pos1>[a-z])
  (?P<num3>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<pos2>[a-z])
""", re.X | re.I)

fmt_normal2 =re.compile(r"""
  ^
  (?P<pre0>[a-z]{3})
  (?P<num0>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<num1>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<keyword>.+) # キーワード
  (?P<num2>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<num3>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<num4>[2-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|viii|vii|vi|v|iv|iii|ii)
  (?P<pos0>[a-z]{2})
""", re.X | re.I)


fmt_jojo = re.compile(r"""
  ^
  (?P<pre0>[a-z])
  (?P<num0>[0-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|ee|ur|ve|en|ht|ne|one|zero|zer|on|ze|ro|ero|viii|vii|vi|v|iv|iii|ii|i)
  (?P<pre1>[a-z])
  (?P<num1>[0-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|ee|ur|ve|en|ht|ne|one|zero|zer|on|ze|ro|ero|viii|vii|vi|v|iv|iii|ii|i)
  (?P<keyword>.+) # キーワード
  (?P<pos0>[a-z])
  (?P<num2>[0-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|ee|ur|ve|en|ht|ne|one|zero|zer|on|ze|ro|ero|viii|vii|vi|v|iv|iii|ii|i)
  (?P<pos1>[a-z]{2})
""", re.X | re.I) 

fmt_intel = re.compile(r"""
  ^
  (?P<pre0>[a-z]{8})
  (?P<num0>[0-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|ee|ur|ve|en|ht|ne|one|zero|zer|on|ze|ro|ero|viii|vii|vi|v|iv|iii|ii|i)
  (?P<keyword>.+) # キーワード
  (?P<num1>[0-9]|three|four|five|seven|eight|nine|thr|two|fou|six|fiv|sev|eig|nin|tw|th|fo|fi|si|se|ei|ni|ee|ur|ve|en|ht|ne|one|zero|zer|on|ze|ro|ero|viii|vii|vi|v|iv|iii|ii|i)
""", re.X | re.I) 

#変換用
dict_num={'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9',
          'three':'3','four':'4','five':'5','seven':'7','eight':'8','nine':'9',
          'thr':'3','two':'2','fou':'4','six':'6','fiv':'5','sev':'7','eig':'8','nin':'9',
          'tw':'2','th':'3','fo':'4','fi':'5','si':'6','se':'7','ei':'8','ni':'9',
          'ee':'3','ur':'4','ve':'5','en':'7','ht':'8','ne':'9','one':'1','zero':'0',
          'zer':'0','on':'1','ze':'0','ro':'0','ero':'0',
          'viii':'8','vii':'7','vi':'6','v':'5','iv':'4','iii':'3','ii':'2','i':'1'}

# 別で使う用
fmt_b32 = re.compile(r""" ^[2-7A-Z]+$""",re.X)
fmt_b64 = re.compile(r""" ^[0-9a-zA-Z\+\/]+$""",re.X)
fmt_bin = re.compile(r""" ^[01]+$""", re.X)
#-----------------------------------------
def is_pass(str, mode):
	'''
		mode: 'normal', 'normal2', 'jojo', 'intel'
	'''
	str=str.lower()
	# ----------------------------------
	# 旧フォーマット #xxx# keyword x#x#x
	if mode=='normal':
		m=fmt_normal.search(str)
		if m:
			result = (
				 dict_num[m.group('num0'   )]
						 +m.group('pre0'   )
				+dict_num[m.group('num1'   )]
						 +m.group('keyword')
						 +m.group('pos0'   )
				+dict_num[m.group('num2'   )]
						 +m.group('pos1'   )
				+dict_num[m.group('num3'   )]
						 +m.group('pos2'   )
			)
			if m.group('keyword') in di.kw:
				return "hit", result
			return "near", result
	# ----------------------------------
	# 新フォーマット xxx## keyword ###xx
	if mode=='normal2':
		m=fmt_normal2.search(str)
		if m:
			result = (
				          m.group('pre0'   )
				+dict_num[m.group('num0'   )]
				+dict_num[m.group('num1'   )]
				         +m.group('keyword')
				+dict_num[m.group('num2'   )]
				+dict_num[m.group('num3'   )]
				+dict_num[m.group('num4'   )]
				         +m.group('pos0'   )
			)
			if m.group('keyword') in di.kw:
				return "hit", result
			return "near", result
	# ----------------------------------
	# JoJo's Word of the Day
	if mode=='jojo':
		m=fmt_jojo.search(str)
		if m:
			result = (
				          m.group('pre0'   )
				+dict_num[m.group('num0'   )]
				         +m.group('pre1'   )
				+dict_num[m.group('num1'   )]
				         +m.group('keyword')
				         +m.group('pos0'   )
				+dict_num[m.group('num2'   )]
				         +m.group('pos1'   )
			)
			if m.group('keyword') in di.jojo:
				return "hit", result
			return "near", result
	# ----------------------------------
	# Anormaly フォーマット
	if mode=='intel':
		m=fmt_intel.search(str)
		if m:
			result = (
				          m.group('pre0'   )
				+dict_num[m.group('num0'   )]
				         +m.group('keyword')
				+dict_num[m.group('num1'   )]
			)
			if m.group('keyword') in di.kw:
				return "hit", result
			return "near", result
	
	return "invalid", str

#-----------------------------------------
# 単純置換用パターン
pattern_morse_atbash = str.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                     'nj*wtqu*mbryiasxfkoeg*dpl*NJ*WTQU*MBRYIASXFKOEG*DPL*5678901234')

pattern_atbash       = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                                     'ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba0987654321')

pattern_hexat        = str.maketrans('0123456789abcdef',
                                     'fedcba9876543210')

pattern_num_atbash   = str.maketrans('123456789',
                                     '987654321')

pattern_key_to_num   = str.maketrans('!@#$%^&*()',
                                     '1234567890')

# 置換処理
def atbash(str):
	return str.translate(pattern_atbash)

def matbash(str):
	'''
		Morse Atbash
	''' 
	return str.translate(pattern_morse_atbash)

def hexat(str):
	'''
		Hex string Atbash
	'''
	str=str.lower()
	return str.translate(pattern_hexat)
def natbash(str):
	'''
		Number atbash
	'''
	return str.translate(pattern_num_atbash)
def key2n(str):
	'''
		keyboard to number
	'''
	return str.translate(pattern_key_to_num)
	
#-----------------------------------------
# 繰り返し置換用パターン (m1付きは-1)

pattern_keyrot1     = str.maketrans('abcdefghijklmnopqrstuvwxyz1234567890;/.,',
                                     ';vxswdfguhjknbiopearycqzt/0123456789l.,m')

pattern_keyrotm1    = str.maketrans(';vxswdfguhjknbiopearycqzt/0123456789l.,m',
                                     'abcdefghijklmnopqrstuvwxyz1234567890;/.,')
                                     
pattern_rot1        = str.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
                                    'bcdefghijklmnopqrstuvwxyzaBCDEFGHIJKLMNOPQRSTUVWXYZA2345678901')

pattern_rotm1       = str.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
                                    'zabcdefghijklmnopqrstuvwxyZABCDEFGHIJKLMNOPQRSTUVWXY0123456789')
# 繰り返し置換処理
def n_substitute(s, n, p, mp):
	b=s
	if(n>=0):
		for i in range(n):
			r = b.translate(p)
			b = r
	else:
		for i in range(-1*n):
			r = b.translate(mp)
			b = r
	return r

# rot cipher
def rot(s, n):
	return n_substitute(s, n, pattern_rot1, pattern_rotm1)
	
# keyboard rot
def kr(s, n):
	s=s.lower()
	return n_substitute(s, n, pattern_keyrot1, pattern_keyrotm1)

#-----------------------------------------
# Vigenere

# 1文字部分
def shift(char, key, chars, decode = True):
	if not char in chars:
		return ''
	if decode:
		return chars[(chars.index(char) - chars.index(key)) % len(chars)]
	else:
		return chars[(chars.index(char) + chars.index(key)) % len(chars)]
# charsを変えれば、同様のことも可能
def vige(char, key, decode=True, chars='abcdefghijklmnopqrstuvwxyz'):
	'''
		vigenere : charsは他のテーブルでやりたい場合に変える
	'''
	if chars.islower():
		char  = char.lower()
		key   = key.lower()
	if chars.isupper():
		char  = char.upper()
		key   = key.upper()
	return ''.join([shift(char[i], key[i%len(key)], chars, decode) for i in range(len(char))])

#-----------------------------------------
def base_to_str(s, base, spt):
	'''
		base変換
		base:基数, spt:デリミタ
	'''
	S= s.split(spt)
	print(S)
	r = list(map(lambda x: int(x, base), S))
	if(max(r)<26):
		c = list(map(lambda x: chr(x+65), r) )
		return ''.join(c)
	if(min(r)>31) and (max(r)<127):
		c = list(map(lambda x: chr(x), r))
		return ''.join(c)
	return 'invalid'
#-----------------------------------------
def split_n(text,n):
	'''
		n文字区切り
		return は、リスト
	'''
	return [text[i*n:i*n+n] for i in range(ceil(len(text)/n))]
#-----------------------------------------
def is_hex(val):
	'''
		Hex形式化どうか
	'''
	try:
		int(val, 16)
		return True
	except ValueError:
		return False
#-----------------------------------------
# Binary エンコード
def b64d(s):
	if len(s)%4 !=1:
		try:
			return base64.b64decode(s+'='*(-len(s)%4)).decode('utf-8')
		except UnicodeDecodeError:
			return 'invalid'
	return 'invalid'
	
def b64e(s):
	return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def b32d(s):
	if len(s)%8 in [0,2,4,5,7]:
		try:
			return base64.b32decode(s+'='*(-len(s)%8)).decode('utf-8')
		except UnicodeDecodeError:
			return 'invalid'
	return 'invalid'
def b32e(s):
	return base64.b32encode(s.encode('utf-8')).decode('utf-8')

# 相互変換
def b32tob64(s):
	if len(s)%8 in [0,2,4,5,7]:
		return base64.b64encode( base64.b32decode(s+'='*(-len(s)%8))).decode('utf-8')
	return 'invalid'
	
def b64tob32(s):
	if len(s)%4 !=1:
		return base64.b32encode( base64.b64decode(s+'='*(-len(s)%4))).decode('utf-8')
	return 'invalid'
#-----------------------------------------
# バイナリ文字列処理用
def bin2ascii(s):
	if len(s)%8==0 and fmt_bin.search(s):
		return ''.join([chr(int(x, 2)) for x in split_n(s, 8)])
	else:
		return 'invalid'

def bin2hex(s):
	if len(s)%8==0 and fmt_bin.search(s):
		return ''.join([format(int(x,2), 'x') for x in split_n(s, 8)])
	else:
		return 'invalid'

def morse2bin(s):
	for x in s:
		if x not in di.dict_to_morse:
			return 'invalid'
	return ''.join([di.dict_to_morse[x] for x in split_n(s, 1)])

def braille2bin(s):
	for x in s:
		if x not in di.dict_to_braille:
			return 'invalid'
	return ''.join([di.dict_to_braille[x] for x in split_n(s, 1)])

def bin2braille(s):
	if len(s)%6==0:
		for x in split_n(s, 6):
			if x not in di.dict_from_braille:
				return 'invalid'
		return ''.join([di.dict_from_braille[x] for x in split_n(s, 6)])
	return 'invalid'
#-----------------------------------------
def rle(s):
	c = ''
	n = 1
	r = ''
	for x in s:
		if c==x:
			n+=1
		elif c =='':
			c=x
			n=1
		else:
			r+=format(n,'x')
			c=x
			n=1
	r+=format(n,'x')
	return r

#-----------------------------------------
#
scoring_init=False
wordlist =[]
def scoring(text):
	'''
		スコアリング
	'''
	global scoring_init
	global wordlist
	n=0
	if not scoring_init:
		wordlist = di.word+di.kw+di.jojo+di.nato
		wordlist = list(set(wordlist))
		scoring_init = True
		
	for w in wordlist:
		if(len(w)>4):
			n+=text.count(w)*len(w)
	return n/len(text)
#-----------------------------------------
def skip_v1(text, n):
	r=''
	for i in range(len(text)):
		index=(i*n)%len(text)
		r+=text[index]
	return r

def skip_v2(text, n):
	r=''
	for i in range(n):
		for j in range(ceil(len(text)/n)):
			index=i+j*n
			if(index<len(text)):
				r+=text[index]
	return r

def rail_fence(text, n):
	output = [""]*len(text)
	pos = 0
	for i in range(n):
		step1, step2 = (n-1-i)*2, i*2
		while i < len(text):
			if step1 != 0:
				output[i], pos, i = text[pos], pos+1, i+step1
			if step2 != 0 and i < len(text):
				output[i], pos, i = text[pos], pos+1, i+step2
	#print("".join(output))
	return "".join(output)
#-----------------------------------------
'''
class Matrix_Walker(Object):
	def __init__(self, mx, my): # matrixサイズ指定
		self.matrix = [[0 for j in range(
	
	def set(self, x, y, rot, dir):
		self.x, self.y, self.rot, self.dir = x, y, rot, dir
		
	def next:
		if rot==0: # 時計周り
			
		else: # 反時計周り
'''
#-----------------------------------------
def rect(text, n, rev_row_odd, rev_row_even, rev_col_odd, rev_col_even):
	'''
		n:幅
		rev_*: 反転フラグ
	'''
	a=split_n(text, n) # 分割
	
	for i in range(len(a)): # 行リバース処理
		if (i%2 ==1 and rev_row_even)or(i%2 ==0 and rev_row_odd):
			a[i]=a[i][::-1]
	
	r=[]
	for i in range(n): # 結果取り出し
		s=''
		for j in range(len(a)):
			s+=a[j][i]
		r.append(s)
	
	for i in range(len(r)): # 縦読みリバース
		if (i%2 ==1 and rev_col_even)or(i%2 ==0 and rev_col_odd):
			r[i]=r[i][::-1]

	return ''.join(r)
#-----------------------------------------
# ヘルプ代わりに関数リストの表示
def deflist():
	i=0
	for x in deflist_glb:
		if i%10==9:
			print(' ',x)
		else:
			print(' ',x,end="")
		i+=1
	print(' ')
#-----------------------------------------
def divisor(n):
	'''
	約数計算
	'''
	large_divisors = []
	for i in range(2, int(sqrt(n) + 1)):
		if n % i == 0:
			yield i
			if i*i != n:
				large_divisors.append(int(n / i))
	for divisor in reversed(large_divisors):
		yield divisor
#-----------------------------------------
# RSA用
def egcd(a, b):
	if a == 0:
		return [b, 0, 1]
	g, y, x = egcd(b%a,a)
	return [g, x - (b//a) * y, y]
def modinv(a, m):
	g, x, y = egcd(a, m)
	if g != 1:
		raise Exception('No modular inverse')
	return x%m
	
def isqrt(n):
  x = n
  y = (x + n // x) // 2
  while y < x:
    x = y
    y = (x + n // x) // 2
  return x
  
  # 基本的には、2つの素数が近い場合のアタック
  # fermat(n)で、p, qが求まる。
def fermat(n):
	x = isqrt(n) + 1
	y = isqrt(x * x - n)

	while True:
		w = x * x - n - y * y
		if w == 0:
			break
		elif w > 0:
			y += 1
		else:
			x += 1
	return x+y, x-y
#-----------------------------------------
#
def md5attack(s):
	code= []
	dict= {}
	word= luc+dig
	r   = ''
	if is_hex(s) and len(s)%32==0:
		code=split_n(s, 32)
		import hashlib as h
		from itertools import permutations as pmu
		
		if not scoring_init:
			wordlist = di.word+di.kw+di.jojo+di.nato
			wordlist = list(set(wordlist))
		
		for x in wordlist:
			y=x.upper()
			dict[h.md5( x.encode('utf-8')).hexdigest()]=x
			dict[h.md5( y.encode('utf-8')).hexdigest()]=y
			
		for x in word:
			y=x.upper()
			dict[h.md5( x.encode('utf-8')).hexdigest()]=x
			dict[h.md5( y.encode('utf-8')).hexdigest()]=y
		
		for x, y in pmu(word, 2):
			#print(h.md5( (x+y).encode('utf-8')).hexdigest())
			dict[h.md5( (x+y).encode('utf-8')).hexdigest()]=x+y
		
		for x, y, z in pmu(word, 3):
			#print(h.md5( (x+y+z).encode('utf-8')).hexdigest())
			dict[h.md5( (x+y+z).encode('utf-8')).hexdigest()]=x+y+z
		
		for x in code:
			if x in dict:
				r+=dict[x]
			else:
				r+='?'
	return r

#-----------------------------------------
# パスコードチェック（引数は、辞書)

def chk(d , mode='normal2'):
	'''
		d: 辞書
		mode: 'normal'/'jojo'/'intel'
	'''
	a=[]
	for s in list(d.keys()):
		history=d[s].split('_')
		if mode=='jojo':
			a.append([is_pass(s, mode),history[0],':'.join(history[1:]),len(history)-1,s,scoring(s)])
		else:
			a.append([is_pass(s, mode),history[0],':'.join(history[1:]),len(history)-1,s])
	return a
#-----------------------------------------
#メインコード
if __name__ == '__main__':

	# 取得
	src  = []
	if(len(sys.argv)>1):
		src=sys.argv[1:]
	else:
		for line in sys.stdin:
			if(len(line)<2):
				break
			src.append(line[:-1])
	# 辞書にしておく
	code = {}
	mode = 'normal2'
	n_loop = 4
	disp_score = False
	
	for a in src:
		if a=='jojo':
			mode='jojo'
			print('--jojo mode--')
		elif a=='jojos':
			mode='jojo'
			n_loop =2
			disp_score=True
			print('--jojo serch mode--')
		elif a=='intel':
			mode='intel'
			print('--intel mode--')
		elif a=='normal':
			mode='normal'
			print('-- normal mode--')
		else:
			code[a]='org'+str(src.index(a))

	# 結果箱	
	r=[]
	r+=(chk(code, mode))

	# 処理ループ
	for i in range(n_loop):
		start = t.time()
		code_new={}
		for s in list(code.keys()):
			
			# 履歴管理用
			hist=code[s]
			hists=hist.split('_')

			# ----------条件なし
			if 'rev' not in hists:
				code_new[s[::-1]]   =hist+'_rev' # reverse
			
			if 'b64e' not in hists and  hists[-1] != 'b64d':
				code_new[b64e(s)]=hist+'_b64e'
			if 'b32e' not in hists and  hists[-1] != 'b32d':
				code_new[b32e(s)]=hist+'_b32e'

			if i==0:
				for n in range(2,int(len(s))): # rail_fence
					code_new[rail_fence(s, n)]=hist+'_rail'+str(n)
				
				for n in range(2,int(len(s))):
					code_new[skip_v1(s, n)]=hist+'_skipv1('+str(n)+')'
					code_new[skip_v2(s, n)]=hist+'_skipv2('+str(n)+')'
			
				for n in divisor(len(s)):
					for b in range(16):
						code_new[rect(s, n, b&8, b&4, b&2, b&1)]=hist+'_rect'+str(n)+'x'+str(int(len(s)/n))
				
				for n in range(2, 25):
					code_new[rot(s,    n)]=hist+'_rot('+str(   n)+')'
					code_new[rot(s, -1*n)]=hist+'_rot('+str(-1*n)+')'
				
				code_new[morse2bin(s.lower())]=hist+'_morse2bin'
				code_new[braille2bin(s.lower())]=hist+'_braille2bin'
				code_new[rle(s)]=hist+'_rle'
				
			# 条件あり
			if s.isalnum(): # アルファベット or 数字
				if 'atb' not in hists:
					code_new[atbash(s)] =hist+'_atb'
				if 'matb' not in hists:
					code_new[matbash(s)]=hist+'_matb'
			if fmt_b64.search(s):
				if 'b64d' not in hists and hists[-1]!='b64e':
					code_new[b64d(s)]=hist+'_b64d'
				if 'b64tob32' not in hists:
					code_new[b64tob32(s)]=hist+'_b64tob32'

			if len(s)>4 and fmt_b32.search(s) and 'b32d' not in hists and hists[-1]!='b32e':
				code_new[b32d(s)]=hist+'_b32d'

			if len(s)>8 and fmt_b32.search(s) and 'b32tob64' not in hists:
				code_new[b32tob64(s)]=hist+'_b32tob64'

			if s.isdigit(): # 10進数字
				if 'numatb' not in hists:
					code_new[natbash(s)]=hist+'_numatb'
				if 'd2c' not in hists:
					code_new[''.join([chr(int(n)) for n in split_n(s,2)]) ]=hist+'_d2c' #2桁Ascii
					code_new[''.join([chr(int(n)+65) for n in split_n(s,2)]) ]=hist+'_d2c' #2桁 num2ascii
					code_new[''.join([chr(int(n)) for n in split_n(s,3)]) ]=hist+'_d2c' #3桁Ascii
			if is_hex(s): # 16進
				if 'hexat' not in hists:
					code_new[hexat(s)]=hist+'_hexat'
				
			if not s.isalnum(): # アルファベット or 数字
				if 'k2n' not in hists:
					code_new[key2n(s)]  =hist+'_k2n'
					
			if len(s)%8==0 and fmt_bin.search(s):
				if 'bin2ascii' not in hists:
					code_new[bin2ascii(s)]=hist+'_bin2ascii'
				if 'bin2hex' not in hists:
					code_new[bin2hex(s)]=hist+'_bin2hex'

			if len(s)%6==0 and fmt_bin.search(s):
				if 'bin2braille' not in hists:
					code_new[bin2braille(s)]=hist+'_bin2braille'
			if fmt_bin.search(s):
				if 'rle' not in hists:
					code_new[rle(s)]=hist+'_rle'
			
		# 重複削除
		for key in list(code_new):
			if key in code:
				del code_new[key]
		# チェック(新規分のみ）
		r+=(chk(code_new, mode))
		if len(code_new)!=0:
			print('comibi = {0:7d},  proc: {1:6.2f}[s] ({2:5.4f}[ms/n])'.format(len(list(code_new.keys()) ),t.time()-start,(t.time()-start)/len(list(code_new.keys()))*1000))
		
		# マージ
		code.update(code_new)
		code_new.clear()
		
	# 結果表示
	from operator import itemgetter
	if disp_score:
		r.sort(key=itemgetter(5))
		
		for result in r:
			print('{0:20s}: {2:7.5f} :{1:s}'.format(result[2], result[4], result[5]) )
	else:
		r.sort(key=itemgetter(1))
	
		for result in r:
			if result[0][0] == 'near':
				#print([result])
				print('{0:8s}:{1:5s}:{2:30s}: {3:s}'.format(result[0][0],result[1],result[2], result[0][1]) )
		for result in r:
			if result[0][0] == 'hit':
				print('{0:8s}:{1:5s}:{2:30s}: {3:s}'.format(result[0][0],result[1],result[2], result[0][1]) )
	
	
#-----------------------------------------以下はただのメモ
'''
islower() 	大文字が含まれていない
isupper() 	小文字が含まれていない
isalpha() 	アルファベットのみ
isdigit() 	数字のみ
isalnum() 	英数字のみ
abcdefghijklmnopqrstuvwxyz0123456789
48cd3f6hijk1mn0pqr57uvwxy2olzeasgtb9
'''
# 多分最後に書いてくと、全ての関数とか取り出せる
deflist_glb=dir()
