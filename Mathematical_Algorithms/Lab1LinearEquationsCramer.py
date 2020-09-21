from numpy import linalg

A=[[3,1,-1,2],[-5,1,3,-4],[2,0,1,-1],[1,-5,3,-3]]
B=[6,-12,1,3]
C=[[3,1,-1,2],[-5,1,3,-4],[2,0,1,-1],[1,-5,3,-3]]
X=[]
for i in range(0,len(B)):
    for j in range(0,len(B)):
        C[j][i]=B[j]
        if i>0:
            C[j][i-1]=A[j][i-1]
    X.append(round(linalg.det(C)/linalg.det(A),1))

print('w=%s'%X[0],'x=%s'%X[1],'y=%s'%X[2],'z=%s'%X[3])
