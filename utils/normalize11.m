function res=normalize11(x,minv,maxv)


res= -1+2*(x-repmat(minv,[size(x,1),1]))./repmat(maxv-minv,[size(x,1),1]);

