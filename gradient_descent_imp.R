args = commandArgs(trailingOnly = TRUE)

# column of Y, any number of predictors >= 1
Y = read.table(args[1])
X = read.table(args[2], header = T)

params = ncol(X) + 1
theta_initial = 1
theta = data.frame(thetas = rep(theta_initial, params))

# cost function
ComputeCost = function(X, Y, theta){
  Yhat = data.frame(rep(theta[1,], nrow(X)))
  for (i in 2:params){
    Yhat = cbind(Yhat, theta[i,]*X[,i-1])
  }
  Yhat = as.data.frame(rowSums(Yhat))
  cost = sum((Y - Yhat)**2/nrow(X))
  return(cost)
}

# gradient 
ComputeGrad = function(X, Y, theta){
  Yhat = data.frame(rep(theta[1,], nrow(X)))
  for (i in 2:params){
    Yhat = cbind(Yhat, theta[i,]*X[,i-1])
  }
  Yhat = as.data.frame(rowSums(Yhat))
  
  grad = matrix(0, nrow = nrow(theta), ncol = 1)
  for (i in 1:nrow(theta)){
    if (i == 1){grad[i] = -2*sum(Y-Yhat)/nrow(X)} else {
      grad[i] = -2*sum(X[,i-1]*(Y-Yhat))/nrow(X)}
  }
  return(grad)
}

# minimize cost wrt eps
cost = ComputeCost(X, Y, theta)
deltaJ = 1
alpha = as.numeric(args[3])
eps = as.numeric(args[4])
while(deltaJ > eps){
  grad = ComputeGrad(X, Y, theta)
  theta = theta - grad*alpha
  prevCost = cost
  cost = ComputeCost(X, Y, theta)
  deltaJ = abs(cost - prevCost)
}

Ypred = data.frame(rep(theta[1,], nrow(X)))
for (i in 2:params){
  Ypred = cbind(Ypred, theta[i,]*X[,i-1])
}
Ypred = rowSums(Ypred)

output_name = args[5]
# file with predicted Y vals
write.table(Ypred, file = output_name, quote = F, col.names = F, row.names = F)




