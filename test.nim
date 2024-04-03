import tables, strutils, strformat, weave

var a = "test".open()
echo type(a)
a.close()
proc task(x: int): int =
  return x

