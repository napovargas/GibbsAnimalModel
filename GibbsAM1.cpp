#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
List GibbsAM(arma::vec y, arma::mat Xin, arma::mat Zin, arma::mat Ain, arma::uword nIter){
  arma::uword       n           = y.n_rows;
  arma::uword       p           = Xin.n_cols;
  arma::uword       q           = Zin.n_cols;
  arma::uword       neq         = p + q;
  wall_clock        timer;
  double            sigma2e     = 1.0;
  double            sigma2a     = 0.5;
  double            pvar        = 1.0;
  double            pmean       = 0.0;
  double            ttime       = 0.0;
  arma::sp_mat      X           = sp_mat(Xin);
  arma::sp_mat      Z           = sp_mat(Zin);
  arma::sp_mat      Ainv        = sp_mat(Ain);
  arma::sp_mat      W           = join_horiz(X, Z);
  arma::sp_mat      WtW         = W.t()*W;
  arma::sp_mat      Sigma       = speye(neq, neq);
  arma::sp_mat      LHS         = WtW + Sigma;
  arma::vec         RHS         = W.t()*y;
  arma::vec         beta        = zeros(p);
  arma::vec         b           = zeros(q);
  arma::vec         eta         = zeros(neq);
  arma::vec         residual    = zeros(n);
  arma::vec         mu          = zeros(n);
  arma::mat         StoreFixed  = zeros(nIter, p);
  arma::mat         StoreRandom = zeros(nIter, q);
  arma::mat         StoreVC     = zeros(nIter, 2);
  List Out;
  
  timer.tic();
  for(uword iter = 0; iter < nIter; iter++){
    LHS         = WtW + Sigma;
    for(uword j = 0; j < neq; j++){
      pvar      = 1.0/LHS(j, j);
      pmean     = pvar*as_scalar(RHS(j) - LHS.row(j)*eta) + eta(j);
      eta(j)    = randn(1)[0]*sqrt(pvar*sigma2e) + pmean; 
    }
    beta        = eta(span(0, p - 1));
    b           = eta(span(p, neq - 1));
    mu          = W*eta;
    residual    = y - mu;
    sigma2e     = 1/randg(1, distr_param(0.5*(n + 10), 1/(0.5*(accu(pow(residual, 2.0)) + 20))))[0];
    sigma2a     = 1/randg(1, distr_param(0.5*(q + 8), 1/(0.5*(as_scalar(b.t()*Ainv*b) + 10))))[0];
    /*
     * Updating Sigma
     */
    Sigma(span(0, p - 1), span(0, p - 1))     = speye(p, p)*sigma2e/1000;
    Sigma(span(p, neq - 1), span(p, neq - 1)) = Ainv*sigma2e/sigma2a;
    /*
     * Storing results
     */
    if(iter%100 == 0){
      Rcpp::Rcout << "Iteration: " << iter + 1 << std::endl;
    }
    StoreFixed.row(iter)  = beta.t();
    StoreRandom.row(iter) = b.t();
    StoreVC(iter, 0)      = sigma2a;
    StoreVC(iter, 1)      = sigma2e;
  }
  ttime                   = timer.toc();
  Out["Fixed"]            = StoreFixed;
  Out["Random"]           = StoreRandom;
  Out["VC"]               = StoreVC;
  Out["time"]             = ttime;
  return(Out);
}

// [[Rcpp::export]]
List GibbsAMRM(arma::vec y, arma::mat Xin, arma::mat Zin, arma::mat Ain, arma::uword nIter){
  arma::uword       n           = y.n_rows;
  arma::uword       p           = Xin.n_cols;
  arma::uword       q           = Zin.n_cols;
  arma::uword       neq         = p + 2*q;
  wall_clock        timer;
  double            sigma2e     = 1.0;
  double            sigma2a     = 0.5;
  double            sigma2p     = 0.75;
  double            pvar        = 1.0;
  double            pmean       = 0.0;
  double            ttime       = 0.0;
  arma::sp_mat      X           = sp_mat(Xin);
  arma::sp_mat      Z           = sp_mat(Zin);
  arma::sp_mat      Ainv        = sp_mat(Ain);
  arma::sp_mat      W           = join_horiz(join_horiz(X, Z), Z);
  arma::sp_mat      WtW         = W.t()*W;
  arma::sp_mat      Sigma       = speye(neq, neq);
  arma::sp_mat      LHS         = WtW + Sigma;
  arma::vec         RHS         = W.t()*y;
  arma::vec         beta        = zeros(p);
  arma::vec         b           = zeros(q);
  arma::vec         pe          = zeros(q);
  arma::vec         eta         = zeros(neq);
  arma::vec         residual    = zeros(n);
  arma::vec         mu          = zeros(n);
  arma::mat         StoreFixed  = zeros(nIter, p);
  arma::mat         StoreRandom = zeros(nIter, q);
  arma::mat         StorePerm   = zeros(nIter, q);
  arma::mat         StoreVC     = zeros(nIter, 3);
  List Out;
  
  timer.tic();
  for(uword iter = 0; iter < nIter; iter++){
    LHS         = WtW + Sigma;
    for(uword j = 0; j < neq; j++){
      pvar      = 1.0/LHS(j, j);
      pmean     = pvar*as_scalar(RHS(j) - LHS.row(j)*eta) + eta(j);
      eta(j)    = randn(1)[0]*sqrt(pvar*sigma2e) + pmean; 
    }
    beta        = eta(span(0, p - 1));
    b           = eta(span(p, q + p - 1));
    pe          = eta(span(q + p, neq - 1));
    mu          = W*eta;
    residual    = y - mu;
    sigma2e     = 1/randg(1, distr_param(0.5*(n + 10), 1/(0.5*(accu(pow(residual, 2.0)) + 20))))[0];
    sigma2a     = 1/randg(1, distr_param(0.5*(q + 8), 1/(0.5*(as_scalar(b.t()*Ainv*b) + 10))))[0];
    sigma2p     = 1/randg(1, distr_param(0.5*(q + 8), 1/(0.5*(accu(pow(pe, 2.0)) + 10))))[0];
    /*
    * Updating Sigma
    */
    Sigma(span(0, p - 1), span(0, p - 1))             = speye(p, p)*sigma2e/1000;
    Sigma(span(p, q + p - 1), span(p, q + p - 1))     = Ainv*sigma2e/sigma2a;
    Sigma(span(q + p, neq - 1), span(q + p, neq - 1)) = speye(q, q)*sigma2e/sigma2p; 
    /*
    * Storing results
    */
    if(iter%100 == 0){
      Rcpp::Rcout << "Iteration: " << iter + 1 << std::endl;
    }
    StoreFixed.row(iter)  = beta.t();
    StoreRandom.row(iter) = b.t();
    StorePerm.row(iter)   = pe.t();
    StoreVC(iter, 0)      = sigma2a;
    StoreVC(iter, 1)      = sigma2e;
    StoreVC(iter, 2)      = sigma2p;
  }
  ttime                   = timer.toc();
  Out["Fixed"]            = StoreFixed;
  Out["Random"]           = StoreRandom;
  Out["Permanent"]        = StorePerm;
  Out["VC"]               = StoreVC;
  Out["time"]             = ttime;
  return(Out);
}

// [[Rcpp::export]]
List GibbsARMT(arma::vec y, arma::mat Xin1, arma::mat Zin1, arma::mat Xin2, arma::mat Zin2, 
               arma::mat Ain, arma::uword nIter){
  arma::uword       n           = y.n_rows;
  arma::uword       n1          = Xin1.n_rows;
  arma::uword       n2          = Xin2.n_rows;
  arma::uword       p1          = Xin1.n_cols;
  arma::uword       p2          = Xin2.n_cols;
  arma::uword       q1          = Zin1.n_cols;
  arma::uword       q2          = Zin2.n_cols;
  arma::uword       neq         = p1 + p2 + q1 + q2;
  wall_clock        timer;
  double            sigma2e     = 1.0;
  double            sigma2a     = 0.5;
  double            sigma2p     = 0.75;
  double            pvar        = 1.0;
  double            pmean       = 0.0;
  double            ttime       = 0.0;
  /*
   * Constructing X, Z and W
   */
  arma::sp_mat      X(n, p1 + p2);
  X(span(0, n1 - 1), span(0, p1 - 1))               = sp_mat(Xin1);
  X(span(n1, n - 1), span(p1, p1 + p2 - 1))         = sp_mat(Xin2);
  arma::sp_mat      Z(n, q1 + 2*q2);
  Z(span(0, n1 - 1), span(0, q1 - 1))               = sp_mat(Zin1);
  Z(span(n1, n - 1), span(q1, q1 + q2 - 1))         = sp_mat(Zin2);
  Z(span(n1, n - 1), span(q1 + q2, q1 + 2*q2 - 1))  = sp_mat(Zin2);
  arma::sp_mat      W                               = join_horiz(X, Z);
  arma::sp_mat      WtW                             = W.t()*W;
  arma::sp_mat      Sigma                           = speye(neq, neq);
  arma::sp_mat      Ainv                            = sp_mat(Ain);
  /*
   * RHS and LHS
   */
  arma::sp_mat      LHS         = WtW + Sigma;
  arma::vec         RHS         = W.t()*y;
  arma::vec         beta        = zeros(p1 + p2);
  arma::vec         b           = zeros(q1 + q2);
  arma::vec         pe          = zeros(q2);
  arma::vec         eta         = zeros(neq);
  arma::vec         residual    = zeros(n);
  arma::vec         mu          = zeros(n);
  arma::mat         StoreFixed  = zeros(nIter, p);
  arma::mat         StoreRandom = zeros(nIter, q);
  arma::mat         StorePerm   = zeros(nIter, q);
  arma::mat         StoreVC     = zeros(nIter, 3);
  List Out;
  
  timer.tic();
  for(uword iter = 0; iter < nIter; iter++){
    LHS         = WtW + Sigma;
    for(uword j = 0; j < neq; j++){
      pvar      = 1.0/LHS(j, j);
      pmean     = pvar*as_scalar(RHS(j) - LHS.row(j)*eta) + eta(j);
      eta(j)    = randn(1)[0]*sqrt(pvar*sigma2e) + pmean; 
    }
    beta        = eta(span(0, p - 1));
    b           = eta(span(p, q + p - 1));
    pe          = eta(span(q + p, neq - 1));
    mu          = W*eta;
    residual    = y - mu;
    sigma2e     = 1/randg(1, distr_param(0.5*(n + 10), 1/(0.5*(accu(pow(residual, 2.0)) + 20))))[0];
    sigma2a     = 1/randg(1, distr_param(0.5*(q + 8), 1/(0.5*(as_scalar(b.t()*Ainv*b) + 10))))[0];
    sigma2p     = 1/randg(1, distr_param(0.5*(q + 8), 1/(0.5*(accu(pow(pe, 2.0)) + 10))))[0];
    /*
    * Updating Sigma
    */
    Sigma(span(0, p - 1), span(0, p - 1))             = speye(p, p)*sigma2e/1000;
    Sigma(span(p, q + p - 1), span(p, q + p - 1))     = Ainv*sigma2e/sigma2a;
    Sigma(span(q + p, neq - 1), span(q + p, neq - 1)) = speye(q, q)*sigma2e/sigma2p; 
    /*
    * Storing results
    */
    if(iter%100 == 0){
      Rcpp::Rcout << "Iteration: " << iter + 1 << std::endl;
    }
    StoreFixed.row(iter)  = beta.t();
    StoreRandom.row(iter) = b.t();
    StorePerm.row(iter)   = pe.t();
    StoreVC(iter, 0)      = sigma2a;
    StoreVC(iter, 1)      = sigma2e;
    StoreVC(iter, 2)      = sigma2p;
  }
  ttime                   = timer.toc();
  Out["Fixed"]            = StoreFixed;
  Out["Random"]           = StoreRandom;
  Out["Permanent"]        = StorePerm;
  Out["VC"]               = StoreVC;
  Out["time"]             = ttime;
  return(Out);
}
