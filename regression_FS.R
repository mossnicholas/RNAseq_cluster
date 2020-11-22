args = commandArgs(trailingOnly = TRUE)

# regression classifier via forward selection
# data - matrix, columns are class data points, rows are features
data = read.table(args[1])
output = args[2]
predictors = rownames(data)
t_data = as.data.frame(t(data))

labels = regmatches(rownames(t_data), regexpr("[^_]*",rownames(t_data)))
labels_b = ifelse(labels==labels[1], 1, 0)
t_data$Y = labels_b
nfold = 5

computeMSE = function(labels, prob){
  MSE = mean((labels - prob)**2)
  return(MSE)
}

# 5-fold CV
compute.cvMSE = function(df, formula, nfold){
  rows = sample(nrow(df))
  shuffled = df[rows,]
  shuffled$fold = rep(c(1:nfold), nrow(t_data)/nfold)
  
  MSEs = rep(NA, nfold)
  for (i in 1:nfold){
    train = shuffled[shuffled$fold != i,]
    test = shuffled[shuffled$fold == i,]
    model.cv = glm(formula, family = "binomial", data = train)
    probs.test = predict.glm(model.cv, type = "response", newdata = test)
    MSEs[i] = computeMSE(test$Y, probs.test)
  }
  return(mean(MSEs))
}

# likelihood estimation
computeLikelihood = function(labels, prob){
  likeli = rep(NA, length(prob))
  for (i in 1:length(prob)){
    likeli[i] = labels[i]*log(prob[i]) + (1-labels[i])*log(1-prob[i])
  }
  return(sum(likeli))
}

# forward selection
results = matrix(NA, nrow = 50, ncol = 4)
added_preds = c()
for (a in 1:50){
  MSEs = c()
  predictors_left = predictors[!predictors %in% added_preds]
  
  for (i in 1:length(predictors_left)){
    if (a==1){
      formula = paste("Y~", predictors_left[i])
    } else {
      initial_form = paste("Y~", paste(added_preds, collapse = "+"))
      formula = as.formula(paste(initial_form, predictors_left[i], sep = "+"))
    }
    model = glm(formula, family = "binomial", data = t_data)
    predicted = predict(model, type = "response")
    error = computeMSE(labels_b, predicted)
    MSEs = c(MSEs, error)
  }
  min = which.min(MSEs)
  added_preds = c(added_preds, predictors_left[min])
  
  if (a==1){
    formula_final = paste("Y~", predictors_left[min])
  } else {
    initial_form = paste("Y~", paste(added_preds, collapse = "+"))
    formula_final = as.formula(paste(initial_form, predictors_left[min], sep = "+"))
  }
  model_final = glm(formula_final, family = "binomial", data = t_data)
  prob_final = predict(model_final, type = "response")
  
  CV = compute.cvMSE(t_data, formula_final, nfold)
  likelihood = computeLikelihood(labels_b, prob_final)
  
  results[a,1] = a
  results[a,2] = predictors_left[min]
  results[a,3] = CV
  results[a,4] = likelihood
}

# summary metrics and ranking of features
colnames(results) = c("Iteration", "Predictor", "CV", "Log-Likelihood")
write.table(results, output, sep = "\t", quote = F, row.names = F)


