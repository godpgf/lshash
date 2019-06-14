from lshash import LSHash
lsh = LSHash([6, 8])
lsh.index([1,2,3,4,5,6,7,8], 'a')
lsh.index([2,3,4,5,6,7,8,9], 'b')
lsh.index([10,12,99,1,5,31,2,3], 'c')
print(lsh.query([1,2,3,4,5,6,7,7], 1)[0][0][1])