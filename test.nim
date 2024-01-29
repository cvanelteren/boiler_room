import tables, strutils, strformat
{.experimental:"codeReordering".}

var a = initTable[int, int]()

a[0] = 3
a[2] =  4
a[-1] = 0
for k, v in a: echo &"{k=} {v=}"
