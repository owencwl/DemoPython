import  thulac

txt1 = open('word.txt', 'r', encoding='utf8').read()
dd=thulac.thulac()

data=dd.cut(txt1,text=False)
print(data)