try
    a = input('Enter the start boundary of the gap: ');
    b = input('Enter the end boundary of the gap: ');
    if (a>=b)
        errordlg('The initial boundary cannot be greater than the final');
        return
    elseif (a<-999999 || a>999999 || b>999999 || b<-999999)
        errordlg('The interval is not within the range. The interval must be between -999999 and 999999');
        return
    endif
catch
  errordlg ('Introduced string');
  return
end_try_catch

try
    h = input('Insert the number of breaking parts: ');
    if (h<=0 || h>999999)
      errordlg ('The number of partitions cannot be less than 1');
      return
    endif 
catch
  errordlg ('Introduced string');
  return    
end_try_catch

n = (b-a)/(h/2)

matrix = zeros(n, 5);

matrix(1,1) = 1
matrix(1,2) = -1
matrix(1,3) = 1

function res = func(z,x)
  res = -z/x;
endfunction

for i = 2:n + 1
    k = zeros(2, 1);
    q = zeros(2, 1);
    matrix(i,1) = (matrix(i-1,1) + h/2);
    q(1) = (func(matrix(i-1,3), matrix(i-1,1)));
    q(2) = (func(matrix(i-1,3) + q(1)*h/2, matrix(i-1,1) + h/2));
    matrix(i,5) = (q(1) + q(2))/2;
    k(1) = matrix(i-1, 3);
    k(2) = (matrix(i-1, 3) + q(1)*h/2);
    matrix(i,4) = (k(1) + k(2))/2;;
    matrix(i,2) = (matrix(i-1,2) + matrix(i,4)*h);
    matrix(i,3) = (matrix(i-1,3) + matrix(i,5)*h);
endfor
matrix
