# Calculate principal components of expression matrix
args = commandArgs(trailingOnly = TRUE)
pca_input = as.matrix(read.table(args[1]))
random_vector = as.vector(read.table(args[2])$V1)

# L2 normalize unit vector
NormVec = function(x) sqrt(sum(x^2))

# Matrix total variance
CalcVar = function(initial, data){
  projected = apply(data, 1, function(x) (x%*%initial)/(initial%*%initial)*initial)
  projected = matrix(unlist(projected), ncol = ncol(data), byrow = TRUE)
  
  total_variance = -sum(apply(projected, 2, function(x) var(x)))
  return(total_variance)
}

# Project matrix onto vector
Project = function(vector, data){
  projected = apply(data, 1, function(x) (x%*%vector)/(vector%*%vector)*vector)
  projected = matrix(unlist(projected), ncol = ncol(data), byrow = TRUE)
  return(projected)
}

vars = c()
components = ncol(pca_input)
pc_start = random_vector
pca_input_start = pca_input

# n x n pc's matrix
pcs = matrix(nrow = components, ncol = components)
for (i in 1:components){
  params = optim(par = pc_start, 
                 fn = CalcVar, 
                 data = pca_input_start)
  pc = params$par
  pc = pc/NormVec(pc)
  vars = c(vars, abs(params$value))
  pcs[,i] = pc
  
  # next iteration pc setup
  pc_start = pc
  projected = Project(pc_start, pca_input_start)
  pca_input_start = pca_input_start - projected
}

df = data.frame(pcs)
rownames(df) = colnames(pca_input)

write.table(vars, paste(args[3], "variance.txt", sep = "_"), quote = F, row.names = F, col.names = F, sep = "\t")
write.table(df, paste(args[3], "PC.txt", sep = "_"), quote = F, sep = "\t")