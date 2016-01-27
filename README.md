# `wvbox`
A wrapper class for creating an interface to word vectors.


##Usage

You can install with `pip install git+https://github.com/lukedeo/wvbox`

```python
from wvbox import WVBox

wv = WVBox('./glove.6B.300d.txt')
wv.build()

docs = ['The dog likes his owner', 'This is an interesting finding']

print wv.get_indices(docs)
# [[1, 828, 1042, 57, 1379], [22, 9, 41, 917, 1482]]
for s in wv.get_words(wv.get_indices(docs)):
	print ' '.join(s)

# the dog likes his owner
# this is an interesting finding
```
