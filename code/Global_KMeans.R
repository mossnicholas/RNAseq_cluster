# Global K-means 
args = commandArgs(trailingOnly = TRUE)
data = read.table(args[1])
output = args[2]

k = 4

# calculate euclidean distance between all data points and centroids
KMeans.cost = function(df, centroids, clusters){
  n = ncol(df)
  k = nrow(centroids)
  distances = matrix(data = NA, nrow = n, ncol = 1)
  for (i in 1:k){
    select.i = which(clusters == i)
    samples = df[,select.i]
    centroid = centroids[i,]
    temp = t(cbind(samples, centroid))
    distance = as.matrix(dist(temp), method = "euclidean")[nrow(temp),]
    distances[select.i,] = distance[c(1:length(select.i))]
  }
  return(mean(distances))
  
}

# global k-means: 
# df = matrix, rows as features and columns as data points
# k = number of clusters
GKMeans = function(df, k){
  p = nrow(df)
  n = ncol(df)
  centroid_start = rowMeans(df)
  
  for (b in 2:k){
    costs = c()
    centroids_iter = c()
    for (a in 1:n){
      centroid = df[,a]
      centroids_all = rbind(centroid_start, centroid)
      
      clusters = matrix(nrow = n, ncol = 1)
      for (i in 1:n){
        temp = rbind(centroids_all, df[,i])
        distance = as.matrix(dist(temp))
        cluster = as.integer(which(distance[b+1,c(1:b)] == min(distance[b+1,c(1:b)])))
        clusters[i,] = cluster
      }
      
      converged = FALSE
      while (!converged){
        previous = clusters
        
        centroids = matrix(nrow = b, ncol = p)
        for (i in 1:b){
          index = which(clusters == i)
          samples = df[,index]
          if (length(index)==1){centroids[i,] = mean(samples)} 
          if (length(index)==0){centroids[i,] = centroids[i,]} 
          if (length(index)>1){centroids[i,] = rowMeans(samples)}
        }
        
        for (i in 1:n){
          temp = rbind(centroids, df[,i])
          distance = as.matrix(dist(temp))
          new_cluster = as.integer(which(distance[b+1,c(1:b)] == min(distance[b+1,c(1:b)])))
          clusters[i,] = new_cluster
        }
        if (all(clusters == previous)){
          converged = TRUE
        }
      }
      final = KMeans.cost(df, centroids, clusters)
      costs = c(costs, final)
      centroids_iter[[a]] = centroids
    }
    cost_min = which(costs == min(costs))[1]
    best_centroid = centroids_iter[[cost_min]]
    centroid_start = best_centroid
  }
  for (i in 1:n){
    temp = rbind(centroid_start, df[,i])
    distance = as.matrix(dist(temp), method = "euclidean")
    new_cluster = as.integer(which(distance[b+1,c(1:b)] == min(distance[b+1,c(1:b)])))
    clusters[i,] = new_cluster
  }
  results = rbind(clusters, min(costs))
  return(results)
}

result = GKMeans(data, k)
write.table(result, output, quote = F, col.names = F, row.names = F)


