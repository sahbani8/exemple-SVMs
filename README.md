exemple-SVMs
============
##SVMs##
## separation lineaire##
n <- 100
p <- 2 
sigma <- 2 # variance of the distribution
meanpos <- 6# centre of the distribution of positive examples
meanneg <- -5 # centre of the distribution of negative examples
npos <- round(n/2) # number of positive examples
nneg <- n-npos # number of negative examples
xpos <- matrix(rnorm(npos*p,mean=meanpos,sd=sigma),npos,p)
xneg <- matrix(rnorm(nneg*p,mean=meanneg,sd=sigma),npos,p)
x <- rbind(xpos,xneg)
y <- matrix(c(rep(1,npos),rep(-1,nneg)))
plot(x,col=ifelse(y>0,1,2))
legend("topleft",c('Positive','Negative'),col=seq(2),pch=1,text.col=seq(2))

ntrain <- round(n*0.6) # number of training examples
tindex <- sample(n,ntrain) # indices of training samples
xtrain <- x[tindex,]
xtest <- x[-tindex,]
ytrain <- y[tindex]
ytest <- y[-tindex]
istrain=rep(0,n)
istrain[tindex]=1

plot(x,col=ifelse(y>0,1,2),pch=ifelse(istrain==1,1,2))
legend("topleft",c('Positive Train','Positive Test','Negative Train','Negative Test'),
col=c(1,1,2,2),pch=c(1,2,1,2),text.col=c(1,1,2,2))

library(kernlab)
svp <- ksvm(xtrain,ytrain,type="C-svc",kernel='vanilladot',C=100,scaled=c())
svp
attributes(svp)
alpha(svp)
alphaindex(svp)
b(svp)
plot(svp,data=xtrain)
ypred = predict(svp,xtest)
table(ytest,ypred)
sum(ypred==ytest)/length(ytest)
ypredscore = predict(svp,xtest,type="decision")
table(ypredscore > 0,ypred)
library(ROCR)
pred <- prediction(ypredscore,ytest)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

perf <- performance(pred, measure = "prec", x.measure = "rec")
plot(perf)

perf <- performance(pred, measure = "acc")
plot(perf)
cv.folds <- function(n,folds=100)
{
split(sample(n),rep(1:folds,length=length(y)))
}
## séparation non-lineaire ## 
svp <- ksvm(x,y,type="C-svc",kernel='rbf',kpar=list(sigma=0.3),C=100)
plot(svp,data=x)
