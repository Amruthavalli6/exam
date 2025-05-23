def is_safe(board,row,col,n):
    for i in range(row):
        if board[i]==col:
            return False
    for i in range(row):
        if abs(board[i]-col)==abs(i-row):
            return False
    return True
def solve_n_queens(board,row,n):
    if row==n:
        return [board[:]]
    solution=[]
    for col in range(n):
        if is_safe(board,row,col,n):
            board[row]=col
            solution+=solve_n_queens(board,row+1,n)
            board[row]=-1
    return solution
def print_n(solutions,n):
    for sol in solutions:
        for row in sol:
            print(" ".join("Q" if i==row else "." for i in range(n)))
        print("--------------------")
n=int(input("enter the number::"))
board=[-1]*n
solutions=solve_n_queens(board,0,n)
print_n(solutions,n)
print('Number of solutions: ',len(solutions))
