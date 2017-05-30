
# understanding what * and ** means in python
def foo(a,b,c,**args):
    print("a=%s"% (a,))
    print("b=%s" % (b,))
    print("c=%s" % (c,))
    print("args=%s" % (args,))


argdict = dict(a="testa", b="testb", c="testc", excessarg="string")
foo(**argdict)


# understand locals()
