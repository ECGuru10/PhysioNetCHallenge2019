function x=nany_na_x(x,q)
    x(isnan(x))=q;
end