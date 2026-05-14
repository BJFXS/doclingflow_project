## Diffusion Adaptation Strategies for Distributed Estimation over Gaussian Markov Random Fields

Paolo Di Lorenzo, Member, IEEE

Department of Information, Electronics, and Telecommunications

'Sapienza' University of Rome, Via Eudossiana 18, 00184 Rome, Italy.

e-mail:

{ dilorenzo } @infocom.uniroma1.it

Abstract -The aim of this paper is to propose diffusion strategies for distributed estimation over adaptive networks, assuming the presence of spatially correlated measurements distributed according to a Gaussian Markov random field (GMRF) model. The proposed methods incorporate prior information about the statistical dependency among observations, while at the same time processing data in real-time and in a fully decentralized manner. A detailed mean-square analysis is carried out in order to prove stability and evaluate the steady-state performance of the proposed strategies. Finally, we also illustrate how the proposed techniques can be easily extended in order to incorporate thresholding operators for sparsity recovery applications. Numerical results show the potential advantages of using such techniques for distributed learning in adaptive networks deployed over GMRF.

on diffusion type of networks. In view of their robustness and adaptation properties, diffusion networks have been applied to model a variety of self-organized behavior encountered in nature, such as birds flying in formation [14], fish foraging for food [15] or bacteria motility [16]. Diffusion adaptation has also been used for distributed optimization and learning [12], to solve dynamic resource allocation problems in cognitive radios [17] and distributed spectrum estimation in small cell networks [18], to perform robust system modeling [19], and to implement distributed learning over mixture models in pattern recognition applications [20].

Index Terms -Distributed LMS estimation, adaptive networks, correlated noise, Gaussian Markov random fields, sparse adaptive estimation, sparse vector.

## I. INTRODUCTION

We consider the problem of distributed estimation [1], where a set of nodes is required to collectively estimate some vector parameter of interest from noisy measurements by relying solely on in-network processing. We consider an ad-hoc network consisting of N nodes that are distributed over some geographic region. At every time instant k , every node i collects a scalar measurement x i [ k ] and a 1 × M regression vector u i [ k ] . The objective is for the nodes in the network to use the collected data { x i [ k ] , u i [ k ] } to estimate some M × 1 parameter vector θ 0 in a distributed manner. There are a couple of distributed strategies that have been developed in the literature for such purposes. One typical strategy is the incremental approach [2]-[5], where each node communicates only with one neighbor at a time over a cyclic path. However, determining a cyclic path that covers all nodes is an NP-hard problem [6] and, in addition, cyclic trajectories are prone to link and node failures. To address these difficulties, consensus based [7] and diffusion-based [8], [9] techniques were proposed and studied in literature. In these implementations, the nodes exchange information locally and cooperate with each other without the need for a central processor. In this way, information is processed on the fly and the data diffuse across the network by means of a real-time sharing mechanism. Since diffusion strategies have shown to be more stable and performing than consensus networks [10], we will focus our attention A characteristic of the observed signal that can be advantageously exploited to improve the estimation accuracy is the sparsity of the parameter to be estimated, i.e., the vector θ 0 contains only a few relatively large coefficients among many negligible ones. Any prior information about the sparsity of θ 0 can be exploited to help improve the estimation performance, as demonstrated in many recent efforts in the area of compressive sensing (CS) [21]-[22]. Up to now, most CS efforts have concentrated on batch recovery methods, where the estimation of the desired vector is achieved from a collection of a fixed number of measurements. In this paper, we are instead interested in adaptive techniques that allow the recovery of sparse vectors to be pursued both recursively and distributively as new data arrive at the nodes. Such schemes are useful in several contexts such as in the analysis of prostate cancer data [23], [24], spectrum sensing in cognitive radio [25],[18], and spectrum estimation in wireless sensor networks [7]. Motivated by the LASSO technique [23] and by connections with compressive sensing [21]-[22], several algorithms for sparse adaptive filtering have been proposed based on Least Mean Squares (LMS) [26]-[27], Recursive Least Squares (RLS) [28], [29], projection-based methods [30], and thresholding operators [31]. A couple of distributed algorithms implementing LASSO over ad-hoc networks have also been considered before, although their main purpose has been to use the network to solve a batch processing problem [24], [32]. One basic idea in all these previously developed sparsity-aware techniques is to introduce a convex penalty term into the cost function to favor sparsity. Our purpose in this work is to use both adaptive and distributed solutions that are able to exploit and track sparsity while at the same time processing data in real-time and in a fully decentralized manner. Doing so would endow networks with learning abilities and would allow them to learn the sparse structure from the incoming data recursively

1

and, therefore, to track variations in the sparsity pattern of the underlying vector as well. Investigations on sparsity-aware, adaptive, and distributed solutions appear in [33]-[38]. In [34][36], the authors employed diffusion techniques that are able to identify and track sparsity over networks in a distributed manner thanks to the use of convex regularization terms. In the related work [33], the authors employ projection techniques onto hyperslabs and weighted ℓ 1 balls to develop a useful sparsity-aware algorithm for distributed learning over diffusion networks. Sparse distributed recursive least squares solutions were also proposed in [37]-[38].

All the previous methods considered the simple situation where the observations are statistically independent. In some circumstances, however, this assumption is unjustified. This is the case, for example, when the sensors monitor a field of spatially correlated values, like a temperature or atmospheric pressure field. In such cases, nearby nodes sense correlated values and then the statistical independence assumption is no longer valid. It is then of interest, in such cases, to check whether the statistical properties of the observations can still induce a structure on the joint probability density function (pdf) that can be exploited to improve network efficiency. There is indeed a broad class of observation models where the joint pdf cannot be factorized into the product of the individual pdf's pertaining to each node, but it can still be factorized into functions of subsets of variables. For instance, this is the case of Markov random fields and Bayes networks [39]. A natural approach in these settings is to incorporate additional prior knowledge in the form of structure and/or sparsity in the relationships among observations. In particular, graphical models provide a very useful method of representing the structure of conditional dependence among random variables through the use of graphs [39]. In the Gaussian case, this structure leads to sparsity in the inverse covariance matrix and allows for efficient implementation of statistical inference algorithms, e.g., belief propagation. Several techniques have been proposed in the literature for covariance estimation, where the structure of the dependency graph is assumed to be known, and covariance selection, where also the structure of the graph is unknown and must be inferred from measurements (see, e.g., [40], [41] and references therein). Recent works on distributed estimation over GMRF appear in [1], [42]-[46].

The contribution of this paper is threefold: (a) The development of novel distributed LMS strategies for adaptive estimation over networks, which are able to exploit prior knowledge regarding the spatial correlation among nodes observations distributed according to a GMRF (To the best of our knowledge this is the first strategy proposed in the literature that exploits the spatial correlation among data in an adaptive and distributed fashion); (b) The derivation of a detailed mean-square analysis that provides closed form expressions for the mean-square deviation (MSD) achieved at convergence by the proposed strategies; (c) The extension of the proposed strategies to include thresholding operators, which endow the algorithms of powerful sparsity recovery capabilities.

The paper is organized as follows. In section II we recall some basic notions from GMRF that will be largely used throughout the paper. In section III we develop diffusion LMS strategies for distributed estimation over adaptive networks, considering spatially correlated observations among nodes. Section IV provides a detailed performance analysis, which includes mean stability and mean-square performance. In Section V, we extend the previous strategies in order to improve performance under sparsity of the vector to be estimated. Section VI provides simulation results in support of the theoretical analysis. Finally, section VII draws some conclusions and possible future developments.

## II. GAUSSIAN MARKOV RANDOM FIELDS

In this section, we briefly recall basic notions from the theory of Gaussian Markov random fields, as this will form the basis of the distributed estimation algorithms developed in the ensuing sections.

A Markov random field is represented through an undirected graph. More specifically, a Markov network consists of [39]:

- 1) An undirected graph G sd = ( V sd , E sd ) , where each vertex v ∈ V sd represents a random variable and each edge { u, v } ∈ E sd represents conditional statistical dependency between the random variables u and v ;
- 2) A set of potential (or compatibility) functions ψ c ( x c ) (also called clique potentials), that associate a nonnegative number to the cliques 1 of G sd .

Let us denote by C the set of all cliques present in the graph. The random vector x is Markovian if its joint pdf admits the following factorization

<!-- formula-not-decoded -->

where x c denotes the vector of variables belonging to the clique c . The functions ψ c ( x c ) are called compatibility functions . The term Z is simply a normalization factor necessary to guarantee that p ( x ) is a valid pdf. A node p is conditionally independent of another node q in the Markov network, given some set S of nodes, if every path from p to q passes through a node in S . Hence, representing a set of random variables by drawing the correspondent Markov graph is a meaningful pictorial way to identify the conditional dependencies occurring across the random variables. If the product in (1) is strictly positive for any x , we can introduce the functions

<!-- formula-not-decoded -->

so that (1) can be rewritten in exponential form as

<!-- formula-not-decoded -->

This distribution is known, in physics, as the Gibbs (or Boltzman) distribution with interaction potentials V c ( x c ) and energy c ∈C V c ( x c ) .

∑ The dependency graph G sd conveys the key probabilistic information through absent edges: If nodes i and j are not neighbors, the random variables x i and x j are statistically independent, conditioned to the other variables. This is the so called pairwise Markov property . Given a subset a ⊂ V sd of vertices, p ( x ) factorizes as

1 A clique is a subset of nodes which are fully connected and maximal, i.e. no additional node can be added to the subset so that the subset remains fully connected.

̸

where the second factor does not depend on a . As a consequence, denoting by S -a the set of all nodes except the nodes in a and by N a the set of neighbors of the nodes in a , p ( x a / x S -a ) reduces to p ( x a / N a ) . Furthermore,

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

This property states that the joint pdf factorizes in terms that contain only variables whose vertices are neighbors. An important example of jointly Markov random variables is the Gaussian Markov Random Field (GMRF), characterized by having a pdf expressed as in (3), with the additional property that the energy function is a quadratic function of the variables. In particular, a vector x of random variables is a GMRF if its joint pdf can be written as

<!-- formula-not-decoded -->

where µ = E { x } is the expected value of x , B = C -1 is the so called precision matrix, with C = E { ( x -µ )( x -µ ) T } denoting the covariance matrix of x . In this case, the Markovianity of x manifests itself through the sparsity of the precision matrix. Indeed, as a particular case of (5), the coefficient b i,j of B is different from zero if and only if nodes i and j are neighbors in the dependency graph, i.e., the corresponding random variables x i and x j are statistically dependent, conditioned to the other variables. The following result from [47] provides an explicit expression between the coefficients of the covariance and the precision matrices for acyclic graphs. The entries of the precision matrix B = { b i,j } , for a positive definite covariance matrix C = { c i,j } and an acyclic dependency graph, are:

<!-- formula-not-decoded -->

 Let us assume that c i,i = σ 2 , for all i , and that the amount of correlation between the neighbors ( i, j ) of the dependency graph is specified by an arbitrary function g ( · ) , which has the Euclidean distance d ij as its argument, i.e. c i,j = σ 2 g ( d ij ) . Furthermore, if we assume that the function g ( · ) is a monotonically non-increasing function of the distance (since amount of correlation usually decays as nodes become farther apart) and g (0) = ν &lt; 1 , exploiting a result from [47], it holds that matrix C is positive definite.

<!-- formula-not-decoded -->

## III. DIFFUSION ADAPTATION OVER GAUSSIAN MARKOV RANDOM FIELDS

Let us consider a network composed of N nodes, where the observation x i [ k ] collected by node i , at time k , follows the linear model

<!-- formula-not-decoded -->

where θ 0 is the M -size column vector to be estimated, and u i [ k ] is a known time-varying regression vector of size M . The observation noise vector v [ k ] = [ v 1 [ k ] , . . . , v N [ k ]] T is distributed according to a Gaussian Markov random field with zero-mean and precision matrix B . Since the vector v is a Gaussian Markov random field, the precision matrix B is typically a sparse matrix that reflects the structure of the dependency graph among the observed random variables.

Following a maximum likelihhod (ML) approach, the optimal estimate for θ 0 can be found as the vector that maximizes the log-likelihood function [48], i.e., as the solution of the following optimization problem:

<!-- formula-not-decoded -->

where p ( x [ k ] , θ ) is the pdf of the observation vector x [ k ] = [ x 1 [ k ] , . . . , x N [ k ]] T collected by all the nodes at time k , which depends on the unknown parameter θ . Since the observation noise is distributed according to a GMRF, exploiting the joint pdf expression (6) and the linear observation model (9), the cooperative estimation problem in (10) is equivalent to the minimization of the following mean-square error cost function:

<!-- formula-not-decoded -->

where U [ k ] = [ u T 1 [ k ] | . . . | u T N [ k ] ] , and we have introduced the weighted norm ‖ y ‖ 2 X = y T Xy , with X denoting a generic positive definite matrix. From (11), we have that the Gaussianity of the observations leads to the mean-square error cost function, while the Markovianity manifests itself through the presence of the sparse precision matrix B . In the case of statistically independent observations, the precision matrix B in (11) becomes a positive diagonal matrix, as already considered in many previous works, e.g., [7]-[9].

For jointly stationary x [ k ] and U [ k ] , if the moments R UB = E { U [ k ] T BU [ k ] } and R UBx = E { U T [ k ] Bx [ k ] } were known, the optimal solution of (11) is given by:

<!-- formula-not-decoded -->

Nevertheless, in many linear regression applications involving online processing of data, the moments R UB and R UBx may be either unavailable or time varying, and thus impossible to update continuously. For this reason, adaptive solutions relying on instantaneous information are usually adopted in order to avoid the need to know the signal statistics beforehand. In general, the minimization of (11) can be computed using a centralized algorithm, which can be run by a fusion center once all nodes transmit their data { x i [ k ] , u i [ k ] } , for all i , to it. A centralized LMS algorithm that attempts to find the solution of problem (11) is given by the recursion

<!-- formula-not-decoded -->

k ≥ 0 , with µ denoting a sufficiently small step-size. Such an approach calls for sufficient communications resources to transmit the data back and forth between the nodes and the central processor, which limits the autonomy of the network, besides adding a critical point of failure in the network due to the presence of a central node. In addition, a centralized solution may limit the ability of the nodes to adapt in realtime to time-varying statistical profiles.

In this paper our emphasis is on a distributed solution, where the nodes estimate the common parameter θ 0 by relying solely on in-network processing through the exchange of data only between neighbors. The interaction among the nodes is modeled as an undirected graph G = ( V c , E c ) and is described by a symmetric N × N adjacency matrix A := { a i,j } , whose entries a i,j are either positive or zero, depending on wether there is a link between nodes i and j or not. To ensure that the data from an arbitrary node can eventually percolate through the entire network, the following is assumed:

Assumption 1 (Connectivity) The network graph is connected; i.e., there exists a (possibly) multihop communication path connecting any two vertexes of the graph .

Due to the presence of the weighted norm in (11) that couples the observations of all the nodes in the network, the problem does not seem to be amenable for a distributed solution. However, since the precision matrix B is a sparse matrix that reflects the structure of the Markov graph, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with x i [ k ] = [ x i [ k ] , { x j [ k ] } j ∈M i ,j&gt;i ] T , and

where M i = { j ∈ V : b i,j &gt; 0 } is the neighborhood of node i in the statistical dependency graph. Thus, exploiting (14) in (11), the global optimization problem becomes

<!-- formula-not-decoded -->

We follow the diffusion adaptation approach proposed in [9], [12], [13], [36], to devise distributed strategies for the minimization of (16). Thus, we introduce two sets of real, weighting coefficients Q = { q j,i } , W = { w j,i } satisfying:

<!-- formula-not-decoded -->

where 1 denotes the N × 1 vector with unit entries and N i denotes the spatial neighborhood of node i . Each coefficient q j,i (and w j,i ) represents a weight value that node i assigns to information arriving from its neighbor j . Of course, the coefficient q j,i (and w j,i ) is equal to zero when nodes j and i are not directly connected. Furthermore, each row of Q adds up to one so that the sum of all weights leaving each node j should be one. Several diffusion strategies can then be derived from (16), see e.g. [9], [12], [13]. For this purpose, we need to explicit the stochastic gradient of each potential function V i ( x i [ k ]; θ ) in (16), which can be written as:

<!-- formula-not-decoded -->

The first algorithm that we consider is the Adapt-ThenCombine (ATC) strategy, which is reported in Table 1. We refer to this strategy as the ATC-GMRF diffusion LMS algorithm. The first step in Table 1 is an adaptation step, where the

## Table 1: ATC-GMRF diffusion LMS

Start with θ i [ -1] and ψ i [ -1] chosen at random for all i . Given non-negative real coefficients { q l,k , w l,k } satisfying (22), and sufficiently small step-sizes µ i &gt; 0 , for each time k ≥ 0 and for each node i , repeat:

<!-- formula-not-decoded -->

intermediate estimate ψ i [ k ] is updated adopting the stochastic gradients of the potential functions V j ( x j [ k ]; θ ) , j ∈ N i , in (15). As we can see from (18), the evaluation of each gradient ∇ θ V i ( x i [ k ]; θ i [ k ]) requires not only measurements from node i , but also data coming from nodes j ∈ M i , j &gt; i , which are neighbors of i in the dependency graph. The coefficients q j,i determine which spatial neighbor nodes j ∈ N i should share its measurements with node i . The second step is a diffusion step where the intermediate estimates ψ j [ k ] , from the spatial neighbors j ∈ N i , are combined through the weighting coefficients { w j,i } . We remark that a similar but

## Table 2: CTA-GMRF diffusion LMS

Start with θ i [ -1] and χ i [ -1] chosen at random for all i . Given non-negative real coefficients { q l,k , w l,k } satisfying (22), and sufficiently small step-sizes µ i &gt; 0 , for each time k ≥ 0 and for each node i , repeat:

<!-- formula-not-decoded -->

alternative strategy, known as the Combine-then-Adapt (CTA) strategy, can also be derived, see, e.g., [9], [12], [13]; in this implementation, the only difference is that data aggregation is performed before adaptation. We refer to this strategy as the CTA-GMRF diffusion LMS algorithm, and we report it in Table 2. The complexity of the GMRF diffusion schemes in (19)-(20) is O (4 M ) , i.e., they have linear complexity as standard stand-alone LMS adaptation.

Remark 1: As we can see from Tables 1 and 2 and eq. (18), the GMRF diffusion LMS algorithms exploit information coming from neighbors defined over two different graphs, i.e., the spatial adjacency graph and the statistical dependency graph. In particular, the algorithms require that: i) each node exchanges information with its neighbors in the Markov dependency graph; ii) the communication graph is connected in order to ensure that the data from an arbitrary sensor can percolate through the entire network. These conditions must be guaranteed by a proper design of the communication graph, which should contain the Markov dependency graph as a subgraph. This represents a generalization of the distributed computation observed in the conditionally independent case, where the exchange of information among nodes takes into account only the spatial proximity of nodes [9]. In the more general Markovian case, the organization of the communication network should take into account, jointly , the grouping suggested by the cliques of the underlying dependency graph.

## IV. MEAN-SQUARE PERFORMANCE ANALYSIS

From now on, we view the estimates θ i [ k ] as realizations of a random process and analyze the performance of the diffusion algorithm over GMRF in terms of its mean-square behavior. Following similar arguments as in [9], we formulate a general form that includes the ATC and CTA algorithms as special cases. Thus, consider a general diffusion filter of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the coefficients p (1) j,i , s j,i , and p (2) j,i are generic nonnegative real coefficients corresponding to the entries of matrices P 1 , S and P 2 , respectively, and satisfy

<!-- formula-not-decoded -->

Equation (21) can be specialized to the ATC-GMRF diffusion LMS algorithm (19) by choosing P 1 = I , S = Q and P 2 = W , and to the CTA-GMRF diffusion LMS algorithm (20) by choosing P 1 = W , S = Q and P 2 = I . To proceed with the analysis, we introduce the error quantities ˜ θ i [ k ] = θ 0 -θ i [ k ] , ˜ χ i [ k ] = θ 0 -χ i [ k ] , ˜ ψ i [ k ] = θ 0 -ψ i [ k ] , and the network vectors:

<!-- formula-not-decoded -->

We also introduce the block diagonal matrix

<!-- formula-not-decoded -->

and the extended block weighting matrices

<!-- formula-not-decoded -->

where ⊗ denotes the Kronecker product operation. We further introduce the random block quantities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, exploiting the linear observation model in (9), we conclude from (18)-(21) that the following relations hold for the error vectors:

<!-- formula-not-decoded -->

We can combine the equations in (28) into a single recursion:

<!-- formula-not-decoded -->

This relation tells us how the network weight-error vector evolves over time. The relation will be the launching point for our mean-square analysis. To proceed, we introduce the following independence assumption on the regression data.

Assumption 2 (Independent regressors) The regressors u i [ k ] are temporally white and spatially independent with R u,i = E { u i [ k ] u T i [ k ] } ≻ 0 .

It follows from Assumption 2 that u i [ k ] is independent of { θ j [ t ] } for all j and t ≤ k -1 . Although not true in general, this assumption is common in the adaptive filtering literature since it helps simplify the analysis. Several studies in the literature, especially on stochastic approximation theory [54]-[55], indicate that the performance expressions obtained using this assumption match well the actual performance of stand-alone filters for sufficiently small step-sizes. Therefore, we shall also rely on the following condition.

Assumption 3 (Small step-sizes) The step-sizes { µ i } are sufficiently small so that terms that depend on higher-order powers of µ i can be ignored.

## A. Convergence in the Mean

Exploiting eq. (26) and Assumption 2, we have

<!-- formula-not-decoded -->

  Then, taking expectations of both sides of (29) and calling upon Assumption 1, we conclude that the mean-error vector evolves according to the following dynamics:

<!-- formula-not-decoded -->

The following theorem guarantees the asymptotic mean stability of the diffusion strategies over GMRF (19)-(20).

Theorem 1 (Stability in the mean) Assume data model (9) and Assumption 2 hold. Then, for any initial condition and any choice of the matrices Q and W satisfying (17), the diffusion strategies (19)-(20) asymptotically converges in the mean if the step-sizes are chosen to satisfy:

<!-- formula-not-decoded -->

where λ max ( X ) denotes the maximum eigenvalue of a Hermitian positive semi-definite matrix X .

<!-- formula-not-decoded -->

## B. Convergence in Mean-Square

We now examine the behavior of the steady-state meansquare deviation, E ‖ ˜ θ i [ k ] ‖ 2 as k →∞ . Following the energy conservation framework of [8], [9] and under Assumption 2, from (29), we can establish the following variance relation:

<!-- formula-not-decoded -->

where Σ is any Hermitian nonnegative-definite matrix that we are free to choose, and

<!-- formula-not-decoded -->

Now, from eq. (27), let us define

<!-- formula-not-decoded -->

where ˆ G = E [ˆ g [ k ]ˆ g T [ k ]] is an MN × MN block matrix, where each block ˆ G i,j is an M × M matrix. Exploiting Assumption 2 and eq. (27), the ( i, i ) -th block of matrix ˆ G , i = 1 , . . . , N , is given by

<!-- formula-not-decoded -->

where the third term in term in (36) can be expressed in closed form. Indeed, defining the set A i = { j ∈ M i , j &gt; i } and associating each term b i,j v j [ k ] , j ∈ Ω i , to the term x t , t = 1 , . . . , m , m = card {A i } , from a direct application of the Multinomial theorem [56], we have

<!-- formula-not-decoded -->

where the products in (37) have only quadratic terms such that

<!-- formula-not-decoded -->

with t, s = 1 , . . . , m , and j, l ∈ A i . At the same way, the ( i, l ) -th block of matrix ˆ G , i, l = 1 , . . . , N , is given by

<!-- formula-not-decoded -->

where I ( Y ) is the indicator function, i.e. I ( Y ) = 1 if the event Y is true and zero otherwise, and A l = { m ∈ M l ; m &gt; l } . Then, given the closed form expression for the matrix G given by eqs. (35)-(40), we can rewrite recursion (33) as:

<!-- formula-not-decoded -->

where Tr( · ) denotes the trace operator. Let σ = vec( Σ ) and σ ′ = vec ( Σ ′ ) , where the vec( · ) notation stacks the columns of Σ on top of each other and vec -1 ( · ) is the inverse operation. Using the Kronecker product property vec( UθV ) = ( V T ⊗ U )vec( Σ ) , we can vectorize both sides of (34) and conclude that (34) can be replaced by the simpler linear vector relation: σ ′ = vec( Σ ′ ) = Fσ , where F is the following N 2 M 2 × N 2 M 2 matrix with block entries of size M 2 × M 2 each:

<!-- formula-not-decoded -->

Using the property Tr( Σ X ) = vec( X T ) T σ we can then rewrite (41) as follows:

<!-- formula-not-decoded -->

The following theorem guarantees the asymptotic mean-square stability (i.e., convergence in the mean and mean-square sense) of the diffusion strategies over GMRF in (19)-(20).

Theorem 2 (Mean-Square Stability) Assume model (9) and Assumption 1 hold. Then, the GMRF diffusion LMS algorithms (19)-(20) will be mean-square stable if the step-sizes are such that (32) is satisfied and the matrix F in (42) is stable.

<!-- formula-not-decoded -->

Remark 2: Note that the step sizes influence (42) through the matrix M . Since in virtue of Assumption 2 the step-sizes are sufficiently small, we can ignore terms that depend on higherorder powers of the step-sizes. Then, we approximate (42) as

<!-- formula-not-decoded -->

where H = ˆ P T 2 ( I -MD ) ˆ P T 1 . Now, since ˆ P 1 and ˆ P 2 are left-stochastic, it can be verified that the above F is stable if I -DM is stable [12], [13]; this latter condition is guaranteed by (32). In summary, sufficiently small step-sizes ensure the stability of the diffusion strategies over GMRF in the mean and mean-square senses.

## C. Mean-Square Performance

Taking the limit as k → ∞ (assuming the step-sizes are small enough to ensure convergence to a steady-state), we deduce from (43) that:

<!-- formula-not-decoded -->

Expression (45) is a useful result: it allows us to derive several performance metrics through the proper selection of the free weighting parameter σ (or Σ ), as was done in [9]. For example, the MSD for any node k is defined as the steadystate value E ‖ ˜ θ i [ k ] ‖ 2 , as k → ∞ , and can be obtained by computing lim k →∞ E ‖ ˜ θ [ k ] ‖ 2 T i with a block weighting matrix T i that has the M × M identity matrix at block ( i, i ) and zeros elsewhere. Then, denoting the vectorized version of the matrix T i by t i = vec(diag( e i ) ⊗ I M ) , where e i is the vector whose i -th entry is one and zeros elsewhere, and if we select σ in (45) as σ i = ( I -F ) -1 t i , we arrive at the MSD for node i :

<!-- formula-not-decoded -->

The average network MSD net is given by:

<!-- formula-not-decoded -->

Then, to obtain the network MSD from (45), the weighting matrix of lim k →∞ E ‖ ˜ θ [ k ] ‖ 2 T should be chosen as T = I MN /N . Let t denote the vectorized version of I MN , i.e., t = vec( I MN ) , and selecting σ in (45) as σ = ( I -F ) -1 t /N , the network MSD is given by:

<!-- formula-not-decoded -->

In the sequel, we will confirm the validity of these theoretical expressions by comparing them with numerical results.

## V. SPARSE DIFFUSION ADAPTATION OVER GAUSSIAN MARKOV RANDOM FIELDS

In this section, we extend the previous algorithms by incorporating thresholding functions that can help improving the performance of the diffusion LMS algorithm over GMRF under a sparsity assumption of the vector θ 0 to be estimated. Since it was argued in [9] that ATC strategies generally outperform CTA strategies, we continue our discussion by focusing on extensions of the ATC algorithm (19); similar arguments applies to CTA strategies. The main idea is to add a sparsification step in the processing chain of the ATC strategy (19), in order to drive the algorithm toward a sparse estimate. In this paper, we consider two main strategies. The first strategy performs the sparsification step after the adaptation and combination steps. We will refer to this strategy as the ACS-GMRF diffusion LMS algorithm, and its main steps are reported in Table 3. The second strategy performs instead the sparsification step in the middle between adaptation and combination steps, as we can notice from Table 4. We will refer to it as the ASC-GMRF diffusion LMS algorithm. The sparsifcation step in Tables 3 and 4 is performed by using a thresholding function T γ ( x ) . Several different functions can

## Table 3: ACS-GMRF diffusion LMS

Start with θ i [ -1] , ψ i [ -1] , ζ i [ -1] chosen at random for all i . Given non-negative real coefficients { q l,k , w l,k } satisfying (22), and sufficiently small step-sizes µ i &gt; 0 , for each time k ≥ 0 and for each node i , repeat:

<!-- formula-not-decoded -->

## Table 4: ASC-GMRF diffusion LMS

Start with θ i [ -1] , ψ i [ -1] , ζ i [ -1] chosen at random for all i . Given non-negative real coefficients { q l,k , w l,k } satisfying (22), and sufficiently small step-sizes µ i &gt; 0 , for each time k ≥ 0 and for each node i , repeat:

<!-- formula-not-decoded -->

be used to enforce sparsity. A commonly used thresholding function comes directly by imposing an ℓ 1 norm constraint in (11), which is commonly known as the LASSO [23]. In this case, the vector threshold function T γ ( x ) is the componentwise thresholding function T γ ( x m ) applied to each element x m of vector x , with

<!-- formula-not-decoded -->

 m = 1 , . . . , M . The function T γ ( x ) in (51) tends to shrink all the components of the vector x and, in particular, attracts to zero the components whose magnitude is within the threshold γ . We denote the strategy using this function as the ℓ 1 -ACSGMRF diffusion LMS algorithm (or its ASC version). Since the LASSO constraint is known for introducing a bias in the estimate, the performance would deteriorate for vectors that are not sufficiently sparse. To reduce the bias introduced by the LASSO constraint, several other thresholding functions can be adopted to improve the performance also in the case of less sparse systems. A potential improvement can be made by modifying the thresholding function T γ ( x ) in (51) as

<!-- formula-not-decoded -->

 m = 1 , . . . , M , where 0 &lt; ε ≪ 1 denotes a small positive weight, f ( y ) = 1 /y , for y ≤ 1 , and f ( y ) = 1 elsewhere.

Compared to (51), the function in (52) adapts the threshold γ · f ( ε + | x m | ) according to the magnitude of the components [51]. When the components are small with respect to ε , the function in (52) increases its threshold so that the components are attracted to zero with a larger probability, whereas, in the case of large components, the threshold is increased to ensure a small effect on them. We denote the strategy using the function in (52) as the reweightedℓ 1 -ACS-GMRF diffusion LMS algorithm (or its ASC version). The reweighted ℓ 1 estimator in (52) is supposed to give better performance than the LASSO. Nevertheless, it still might induce a too large bias if the vector is not sufficiently sparse. To further reduce the effect of the bias, we consider the non-negative GAROTTE estimator as in [52], whose thresholding function is defined as a vector whose entries are derived applying the threshold

<!-- formula-not-decoded -->

 m = 1 , . . . , M . We denote the strategy using the function in (53) as the G-ACS-GMRF diffusion LMS algorithm (or its ASC version). Ideally, sparsity is better represented by the ℓ 0 norm as the regularization factor in (11); this norm denotes the number of non-zero entries in a vector. Considering that ℓ 0 norm minimization is an NP-hard problem, the ℓ 0 norm is generally approximated by a continuous function. A popular approximation [27], [34] is

<!-- formula-not-decoded -->

where β &gt; 0 is a shape parameter. Based on a first order Taylor approximation of (54), the thresholding function associated to the ℓ 0 norm can be expressed as [38]:

<!-- formula-not-decoded -->

m = 1 , . . . , M , with β &lt; √ 1 /γ . We can see how the ℓ 0 thresholding function takes non-uniform effects on different components, and shrinks the small components around zero. We denote the strategy using the function in (55) as the ℓ 0 -ACS-GMRF diffusion LMS algorithm (or its ASC version). In the sequel, numerical results will show the performance achieved by adopting the thresholding functions in (51), (52), (53), and (55).

̸

Remark 3: It is important to highlight the pros and cons of the proposed strategies in (49) and (50). The adoption of the thresholding functions in (51)-(55), determines that, if the vector θ 0 is sparse, after the sparsification step only a subset of the entries of the local estimates are different from zero. Indeed, this thresholding operation allows to estimate the support of the vector θ 0 , i.e., the set of indices of the non-zero component, which is denoted by supp( θ 0 ) = { m : θ 0 ,m = 0 } . Now, since in the ACS strategy in (49) the combination step is performed before the sparsification, the thresholding function will be able to correctly identify the zero entries of the vector with larger probability with respect to the ASC strategy in

(50), thanks to the noise reduction effect due to the cooperation among nodes. At the same time, sparsifying the vector before the combination step, as it is performed in the ASC strategy, has the advantage that, if the vector is very sparse, each node must transmit to its neighbors only the few entries belonging to the estimated vector support, thus remarkably reducing the burden of information exchange. This intuition suggests that the two strategies lead to an interesting tradeoff between performance and communication burden, as we will illustrate in the numerical results.

The following theorem guarantees the asymptotic meansquare stability (i.e., stability in the mean and mean-square sense) of the sparse diffusion strategies over GMRF in (49)(50). Interestingly, stability is guaranteed under the same conditions of the sparsity agnostic strategies in (19)-(20).

Theorem 3 (Mean-Square Stability) Assume model (9) and Assumption 2 hold. Then, the sparse diffusion strategies over GMRF (49)-(50) will be mean-square stable if condition (32) is satisfied and the matrix F in (42) is stable.

Proof: See Appendix C.

## VI. NUMERICAL RESULTS

In this section, we provide some numerical examples to illustrate the performance of the diffusion strategies over GMRF. In the first example, we evaluate the performance of the proposed strategies, comparing it with respect to standard diffusion algorithms from [9]. The second example shows the benefits of using the ACS and ASC strategies in (49)-(50) in the case of sparseness of the vector to be estimated. The third example illustrates the capability of the proposed strategies to track time-varying, sparse vector parameters.

Numerical Example - Performance : We consider a connected network composed of 20 nodes. The spatial topology of the network is depicted in Fig. 1 (all the links are communication links). The regressors u i [ k ] have size M = 10 and are zero-mean white Gaussian distributed with covariance matrices R u,i = σ 2 u,i I M , with σ 2 u,i shown on the bottom side of Fig. 1. The noise variables are assumed to be distributed according to a GMRF, whose statistical dependency graph is depicted through the thick links in Fig. 1. Each thick link is also supported by a communication link so that the dependency graph can be seen as a sub-graph of the communication graph. Since the dependency graph in Fig. 1 is acyclic, we compute the precision matrix as in (7) with c i,i = σ 2 = 0 . 0157 and c i,j = σ 2 ν exp( -κ · d ij ) , where d ij is the Euclidean distance among nodes i and j , ν &lt; 1 is the nugget parameter, and κ ≥ 0 is a correlation coefficient.

In this example, we aim to illustrate the potential gain offered by the proposed strategies in estimating a vector parameter embedded in a GMRF. To this goal, in Fig. 2 we show the learning behavior of 6 different strategies for adaptive filtering: stand alone LMS, CTA and ATC diffusion strategies from [9], the proposed CTA and ATC GMRF diffusion strategies in Tables 1 and 2, and the centralized LMS solution in (13). The parameters of the GMRF are ν = 0 . 9 and κ = 0 . 1 . The step-size of the GMRF diffusion strategies is equal to 3 × 10 -4 , whereas the step-sizes of the other algorithms are chosen in order to have the same convergence rate of the proposed strategies. We consider diffusion algorithms without measurement exchange, i.e. Q = I . Instead, the combination matrix W in (17) for the diffusion strategies is chosen such that each node simply averages the estimates from the neighborhood, i.e., w ij = 1 / |N i | for all i . As we can notice from Fig. 2, thanks to the prior knowledge of the structure of the underlying dependency graph among the observations, the proposed ATC and CTA GMRF diffusion strategies lead to a gain with respect to their agnostic counterparts. The ATC strategy outperforms the CTA strategy, as in the case of standard diffusion LMS [9]. From Fig. 2, we also notice the large gain obtained by the diffusion strategies with respect to stand-alone LMS adaptation. Furthermore, we can see how the performance of the ATC-GMRF diffusion strategy is very close to the LMS centralized solution in (13), which has full knowledge of all the network parameters and observations.

Fig. 1: Adjacency (all the links) and Dependency (thick links) graphs (top), and regressor variances (bottom).

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000000_bd09e99f132d29acc4da176d23e9b6a4c77e71e57fc1d346746c7c17e6c7d63f.png)

Fig. 2: Network MSD versus iteration index, considering different algorithms.

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000001_e2a2ff01bf798938320c89b38e80e5da14494bcaf0ebd58d0c73568a6469b3bf.png)

Fig. 3: MSD versus node index, comparing theoretical results with numerical simulations.

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000002_5e9dde59f214371579560f7bfdc4ff780d11937ee77c6a3be2dec3185ce4ca5e.png)

Fig. 4: MSD gain versus ν , for different values of κ .

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000003_6e1772e3c0adc42c52bc36f8f69a7874a49ed05b3bdaf06c65fc932136818ccc.png)

To check the validity of the theoretical derivations in (46), in Fig. 3 we illustrate the behavior of the steady-state MSD of the ATC and CTA GMRF diffusion strategies, at each node in the network, comparing the theoretical values with simulation results. The MSD values are obtained by averaging over 100 independent simulations and over 200 samples after convergence. From Fig. 3, we can notice the good matching between theory and numerical results.

To assess the sensitivity of the proposed strategies to variations in the parameters describing the GMRF, in Fig. 4, we report the difference in dB between the steady-state network MSD of the ATC (from [9]) and ATC-GMRF (table 1) diffusion algorithms (i.e., the gain in terms of MSD), versus the nugget parameter ν , considering different values of the coefficient κ . The results are averaged over 100 independent realizations and over 200 samples after convergence. The parameters are the same of the previous simulation and, for any pair ( ν, κ ) , the step-sizes of the two algorithms are chosen in order to match their convergence rate. As we can see from Fig. 4, as expected, the MSD gain improves by increasing the correlation among the observations, i.e. by increasing the nugget parameter ν and reducing the coefficient κ .

Fig. 5: MSD versus number of non zero components of θ 0 , considering different algorithms.

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000004_57735a77324a88f35f80435a3580309522812a06d8046c5a7428d9062bef6443.png)

Numerical Example -Sparsity Recovery : This example aims to show the steady-state performance for the sparse GMRF diffusion algorithms, considering the different thresholding functions illustrated in Section V. The regressors u i [ k ] have size M = 50 and are zero-mean white Gaussian distributed with covariance matrices R u,i = σ 2 u,i I M , with σ 2 u,i shown on the bottom side of Fig. 1. In Fig. 5, we report the steady-state network Mean Square Deviation (MSD), versus the number of non-zero components of the true vector parameter (which are set to 1), for 5 different adaptive filters: the ATC-GMRF diffusion described in Table 1 (i.e., the sparsity agnostic GMRF diffusion algorithm), the ℓ 1 -ACS GMRF diffusion, the Rwℓ 1 -ACS GMRF diffusion, the G-ACS GMRF diffusion, and the ℓ 0 -ACS GMRF diffusion, which are described in Table 3 and by (51), (52), (53), and (55), respectively. The results are averaged over 100 independent experiments and over 200 samples after convergence. The step-sizes are chosen as µ i = 2 . 8 × 10 -4 for all i , and the parameters of the GMRF are ν = 0 . 9 and κ = 0 . 1 . The combination matrix W is chosen such that w ij = 1 / |N i | for all i . The threshold parameters of the various strategies are available in Fig. 5. As we can see from Fig. 5, when the vector is very sparse all the sparsity-aware strategies yield better steady-state performance than the sparsity agnostic algorithm. The Rwℓ 1 , the garotte, and the ℓ 0 estimators greatly outperform the lasso thanks to the modified thresholding operations in (52), (53), and (55). When the vector is less sparse, the ℓ 1 -ACS GMRF strategy performs worse than the sparsity agnostic algorithm due to the dominant effect of the bias introduced by the function in (51), whereas the other strategies still lead to a positive gain. In particular, while in this example the Rwℓ 1 -ACS GMRF and the G-ACS GMRF diffusion strategies perform worse than the sparsity agnostic ATC-GMRF diffusion algorithm if the number of non-zero components is larger than 37 and 45, respectively, the ℓ 0 -ACS GMRF strategy leads always to a positive gain, thus matching the performance of the sparsity agnostic strategy only when the vector θ 0 is completely non-sparse.

To compare the performance of the proposed strategies with other distributed, sparsity-aware, adaptive techniques available in the literature, we illustrate the temporal behavior of the network MSD of four adaptive filters: the ℓ 0 -ACS GMRF described in Table 3 and by (55), the ℓ 0 -ASC GMRF described in Table 4 and by (55), the ℓ 0 -ATC sparse diffusion LMS from [34], [36], [18], and the projection based sparse learning from [33]. The results are averaged over 100 independent experiments. We consider a vector parameter θ 0 with only 6 elements set equal to one, which have been randomly chosen. The threshold parameters of the ℓ 0 -ACS GMRF (and ℓ 0 -ASC GMRF) are chosen such that γ = 10 -4 , and β = 50 . The step-sizes, the combination matrix W , and the GMRF parameters are chosen as before. Using the same notation adopted in [34], the parameters of the ℓ 0 Sparse diffusion are ρ = 2 × 10 -3 and α = 5 . Using the same notation adopted in [33], the parameters of the projection based filter are: ε = 1 . 3 × max k ( σ v,i ) ; µ n = 0 . 06 × M n ; the radius of the weighted ℓ 1 ball is equal to ‖ w o ‖ 0 = 6 (i.e., the correct sparsity level); ˜ ε n = 0 . 02 ; α = 0 . 85 for k &lt; 160 and α = 0 . 65 for k &gt; 160 ; the number of hyperslabs used per time update is equal to q = 20 . From Fig. 6, we notice how the ℓ 0 -ACS GMRF algorithm outperforms all the other strategies. This is due to the exploitation of the prior knowledge regarding the underlying GMRF and the adoption of the thresholding function in (55), which gives powerful capabilities of sparsity recovery to the algorithm. As previously intuited in section V, ACS strategies outperform ASC strategies thanks to the exploitation of the cooperation among nodes for noise reduction before the sparsification step. At the same time, since in the ASC implementation each node transmits to its neighbors only the entries belonging to the estimated vector support, the information exchange in the network is greatly reduced. Thus, ACS and ASC strategies constitute an interesting tradeoff between performance and communication burden. These two algorithms have both linear complexity, i.e., O (4 M ) . At the same time, the ℓ 0 ATC sparse diffusion LMS from [34], [36], [18], has a linear complexity too, i.e. O (3 M ) , whereas the projection-based method is more complex, i.e., O ( M (3 + q + log M )) , due to the presence of q projections onto the hyperslabs and 1 projection on the weighted ℓ 1 ball per iteration. This discussion further enlighten the good features of the proposed strategies for distributed, adaptive and sparsity-aware estimation.

Fig. 6: Network MSD versus iteration index, considering different algorithms.

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000005_f74bfdf6ba1eb721b3997af7248505f1431ea4f11764eba6d978230a6197a305.png)

Fig. 7: Example of tracking capability: Temporal behavior of the estimate of the first (top) and twentyfifth (bottom) components of the time-varying vector θ 0 [ k ] .

![Image](/data/output/markdown/pdf_two_column/paper_01/document_artifacts/image_000006_0e1b80a47e160140c75f89c551b17a4648c64dbf10559a319981cee4d4635afb.png)

Numerical Example - Tracking capability : The aim of this example is to illustrate the tracking capability of the proposed strategies. We consider the ℓ 0 -ACS GMRF described in Table 3 and by (55). In this example, the algorithm is employed to track a time-varying parameter that evolves with time as θ 0 [ k ] = 0 . 98 × θ 0 [ k -1] + s [ k ] , where s [ k ] is a Gaussian random variable with mean 0 . 01 × 1 M and covariance matrix 4 × 10 -2 I . In finite time intervals chosen at random, the components of the vector parameter are set to zero. In Fig. 7 we illustrate the behavior of the estimate of the first and twentyfifth components of the time-varying vector θ 0 [ k ] , superimposing also the true behavior of the parameters for comparison purposes. The other parameters are the same of the previous simulation, except for the step-size that is set equal to 10 -3 . As we can notice from Fig. 7, the algorithm tracks quite well the fluctuations of the parameter. Furthermore, thanks to the use of the thresholding function in (55), the algorithm is also able to track sparsity in a very efficient manner, thus setting exactly to zero the vector components that are found smaller than a specific threshold.

## VII. CONCLUSIONS

In this paper we have proposed distributed strategies for the online estimation of vectors over adaptive networks, assuming the presence of spatially correlated measurements distributed according to a GMRF model. The proposed strategies are able to exploit the underlying structure of the statistical dependency graph among the observations collected by the network nodes at different spatial locations. A detailed mean square analysis has been carried out and confirmed by numerical simulations. We have also illustrated how the proposed strategies can be extended by incorporating thresholding functions, which improve the performance of the algorithms under sparsity of the vector parameter to be estimated. Several simulation results illustrate the potential advantages achieved by these strategies for online, distributed, sparse vector recovery.

The proposed methods require the apriori knowledge of the structure of the dependency graph existing among the observations collected at different nodes. In practical applications, this means that the precision matrix must be previously estimated by the sensor network, using a sparse covariance selection method, see, e.g. [41] and references therein. Then, once each node is informed about the local structure of the dependency graph defined by the precision matrix, the network can run the proposed strategies in a fully distributed fashion. An interesting future extension of this work might be to couple the proposed algorithms with (possibly distributed) online methods for covariance selection. In this way a further layer of adaptation would be added into the system, thus enabling the network to track also temporal variations in the spatial correlation among data. This problem will be the tackled in a future publication.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ θ [0] is the initial condition. As long as we can show that H k converge to zero as k goes to infinity, then we would be able to conclude the convergence of E ˜ θ [ k ] . To proceed, we call upon results from [11], [12], [13]. Let z = col { z 1 , z 2 , . . . , z N } denote a vector that is obtained by stacking N subvectors of size M × 1 each (as is the case with ˜ θ [ k ] ). The block maximum norm of z is defined as

<!-- formula-not-decoded -->

where ‖·‖ denotes the Euclidean norm of its vector argument. Likewise, the induced block maximum norm of a block matrix X with M × M block entries is defined as:

̸

<!-- formula-not-decoded -->

Now, since

<!-- formula-not-decoded -->

recursion (56) converges to zero as i →∞ if we can ensure that ‖ H ‖ b, ∞ &lt; 1 . This condition is actually satisfied by (32). To see this, we note that

<!-- formula-not-decoded -->

since ∥ ∥ ∥ ˆ P T 1 ∥ ∥ ∥ b, ∞ = ∥ ∥ ∥ ˆ P T 2 ∥ ∥ ∥ b, ∞ = 1 in view of the fact that ˆ P 1 and ˆ P 2 are left-stochastic matrices [11]. Therefore, to satisfy ‖ H ‖ b, ∞ &lt; 1 , it suffices to require

<!-- formula-not-decoded -->

Now, we recall a result from [12] on the block maximum norm of a block diagonal and Hermitian matrix X with M × M blocks { X i } , which states that

<!-- formula-not-decoded -->

with ρ ( U ) denoting the spectral radius of the Hermitian matrix U . Thus, since M is diagonal, condition (61) will hold if the matrix I -MD is stable. Using (30), we can easily verify that this condition is satisfied for any step-sizes satisfying (32), as claimed before. This concludes the proof of the theorem.

## APPENDIX B

## PROOF OF THEOREM 2

Letting r = vec( ˆ P T 2 MG T M ˆ P 2 ) , recursion (43) leads to:

<!-- formula-not-decoded -->

where E ‖ ˜ θ [0] ‖ 2 is the initial condition. We first note that if F is stable, F k → 0 as k →∞ . In this way, the first term on the RHS of (63) vanishes asymptotically. At the same time, the convergence of the second term on the RHS of (63) depends only on the geometric series of matrices ∑ ∞ l =0 F l , which is known to be convergent to a finite value if the matrix F is a stable matrix [ 58 ] . In summary, since both the first and second terms on the RHS of (63) asymptotically converge to finite values, we conclude that E ‖ ˜ θ [ k ] ‖ 2 σ will converge to a steadystate value, thus completing our proof.

## APPENDIX C

## PROOF OF THEOREM 3

We will carry out the proof for the ACS strategy in (49). The proof for the ASC strategy follows from straightforward modifications. Following the arguments in Section IV, we define the vectors ˜ θ i [ k ] = θ 0 -θ i [ k ] , ˜ ζ i [ k ] = θ 0 -ζ i [ k ] , and the network vectors:

<!-- formula-not-decoded -->

Then, the evolution of the error vector ˜ θ [ k ] can be written as

<!-- formula-not-decoded -->

The thresholding functions in (51)-(55) can be cast as

<!-- formula-not-decoded -->

with c 1 = γ √ M for (51)-(53) and c 1 = γβ √ M for (55). Then, substituting (66) in (65), we have

<!-- formula-not-decoded -->

with H [ k ] = ˆ W T ( I -MD [ k ]) because, for the ACS strategy in (49), we have ˆ P 2 = ˆ W = W ⊗ I N and ˆ P 1 = I . Taking the expectation of both terms in (67) and letting H = E H [ k ] = ˆ W T ( I -MD ) , the recursion can be cast as

<!-- formula-not-decoded -->

Taking the block maximum norm of E ˜ θ [ k ] in (68) and exploiting the boundness of function f ( · ) , we have

<!-- formula-not-decoded -->

where 0 &lt; c 2 &lt; ∞ . The right-hand side of (69) converges as k → ∞ to a fixed value if ‖ H ‖ b, ∞ &lt; 1 . As shown in Appendix A, this condition is verified by choosing the stepsizes in order to satisfy (32). This proves the stability in the mean of the ACS strategy (49).

To prove the stability of the ACS strategy (49) in the meansquare sense, using the same notation of Section IV.B and letting r = vec( ˆ P T 2 MG T M ˆ P 2 ) , we have from (67) that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Since f ( · ) and E ˜ θ [ k ] are bounded by positive constants for any k , we have | f 2 ( ˜ θ [ k -1]) | &lt; c 3 , with 0 &lt; c 3 &lt; ∞ . The positive constant c 3 can be related to the quantity r T σ in (70) through some constant υ ∈ R + , say, c 3 = υ r T σ . Thus, from (70), we can derive the upper bound

<!-- formula-not-decoded -->

which leads to the recursion

<!-- formula-not-decoded -->

where E ‖ ˜ θ [0] ‖ 2 is the initial condition. Using the same arguments as in Appendix B, the right hand side of (73) converges to a fixed value if F is a stable matrix. This proves the boundness of the quantity E ‖ ˜ θ [ k ] ‖ 2 Σ for all k and, ultimately, the mean-square stability of the ACS strategy (49). This concludes the proof of the theorem.

## REFERENCES

- [1] S. Barbarossa, S. Sardellitti, and P. Di Lorenzo, 'Distributed Detection and Estimation in Wireless Sensor Networks,' E-Reference Signal Processing, R. Chellapa and S. Theodoridis, Eds., Elsevier, 2013.
- [2] D. Bertsekas, 'A new class of incremental gradient methods for least squares problems,' SIAM Journal on Optimization , vol. 7, no. 4, pp. 913926, 1997.
- [3] A. Nedic and D. Bertsekas, 'Incremental subgradient methods for nondifferentiable optimization,' SIAM Journal on Optimization , vol. 12, no. 1, pp. 109-138, 2001.
- [4] C. Lopes and A. H. Sayed, 'Incremental adaptive strategies over distributed networks,' IEEE Transactions on Signal Processing , vol. 55, no. 8, pp. 4064-4077, 2007.
- [5] L. Li, J. Chambers, C. Lopes, and A. H. Sayed, 'Distributed estimation over an adaptive incremental network based on the affine projection algorithm,' IEEE Transactions on Signal Processing , vol. 58, no. 1, pp. 151-164, 2009.

- [6] R. M. Karp, 'Reducibility among combinational problems,' Complexity of Computer Computations (R. E. Miller and J. W. Thatcher, eds.), pp. 85-104, 1972.
- [7] I. D. Schizas, G. Mateos, and G. B. Giannakis, 'Distributed LMS for consensus-based in-network adaptive processing,' IEEE Trans. Signal Process. , vol. 57, no. 6, pp. 2365-2382, Jun. 2009.
- [8] C. G. Lopes and A. H. Sayed, 'Diffusion least-mean squares over adaptive networks: Formulation and performance analysis,' IEEE Transactions on Signal Processing , vol. 56, no. 7, pp. 3122-3136, July 2008.
- [9] F. S. Cattivelli and A. H. Sayed, 'Diffusion LMS strategies for distributed estimation,' IEEE Trans. on Signal Proc. , vol. 58, pp. 1035-1048, 2010.
- [10] S.-Y. Tu and A. H. Sayed, 'Diffusion strategies outperform consensus strategies for distributed estimation over adaptive networks,' IEEE Trans. Signal Processing , vol. 60, no. 12, pp. 6217-6234, Dec. 2012.
- [11] N. Takahashi, I. Yamada, and A. H. Sayed, 'Diffusion least-mean squares with adaptive combiners: Formulation and performance analysis,' IEEE Transactions on Signal Processing , vol. 58, no. 9, pp. 4795-4810, Sep. 2010.
- [12] J. Chen and A. H. Sayed, 'Diffusion adaptation strategies for distributed optimization and learning over networks,' IEEE Transactions on Signal Processing , vol. 60, no. 8, pp. 4289-4305, 2012.
- [13] A. H. Sayed, 'Diffusion adaptation over networks,' in Academic Press Library in Signal Processing , vol. 3, R. Chellapa and S. Theodoridis, editors, pp. 323-454, Academic Press, Elsevier, 2013.
- [14] F. Cattivelli and A. H. Sayed, 'Modeling bird flight formations using diffusion adaptation,' IEEE Transactions on Signal Processing , vol. 59, no. 5, pp. 2038-2051, May 2011.
- [15] S-Y. Tu and A. H. Sayed, 'Mobile adaptive networks,' IEEE J. Sel.Topics on Signal Processing , vol. 5, no. 4, pp. 649-664, August 2011.
- [16] J. Chen, X. Zhao, and A. H. Sayed, 'Bacterial motility via diffusion adaptation,' Proc. 44th Asilomar Conference on Signals, Systems and Computers , Pacific Grove, CA, Nov. 2010.
- [17] P. Di Lorenzo, S. Barbarossa, and Ali H. Sayed, 'Bio-Inspired Decentralized Radio Access based on Swarming Mechanisms over Adaptive Networks,' IEEE Transactions on Signal Processing , Vol. 61, no. 12, pp. 3183-3197, 15 June 2013.
- [18] P. Di Lorenzo, S. Barbarossa, and A. H. Sayed, 'Distributed Spectrum Estimation for Small Cell Networks based on Sparse Diffusion Adaptation,' IEEE Signal Processing Letters , Vol. 20, no. 12, pp. 1261-1265, December 2013.
- [19] S. Chouvardas, K. Slavakis, and S. Theodoridis, 'Adaptive robust distributed learning in diffusion sensor networks,' IEEE Transactions on Signal Processing , vol. 59, no. 10, pp. 4692-4707, 2011.
- [20] Z. Towfic, J. Chen and A. H. Sayed, 'Collaborative learning of mixture models using diffusion adaptation,' in Proc. IEEE Workshop on Machine Learning for Signal Processing , Beijing, China, Sept. 2011.
- [21] D. Donoho, 'Compressed sensing,' IEEE Transactions on Information Theory , vol. 52, no. 4, pp. 1289-1306, 2006.
- [22] R. Baraniuk, 'Compressive sensing,' IEEE Signal Processing Magazine , vol. 25, pp. 21-30, March 2007.
- [23] R. Tibshirani, 'Regression shrinkage and selection via the LASSO,' J. Royal Statistical Society: Series B , vol. 58, pp. 267-288, 1996.
- [24] G. Mateos, J. A. Bazerque, and G. B. Giannakis, 'Distributed sparse linear regression,' IEEE Transactions on Signal Processing , vol 58, No. 10, pp. 5262-5276, Oct. 2010.
- [25] J. A. Bazerque, and G. B. Giannakis, 'Distributed spectrum sensing for cognitive radio networks by exploiting sparsity,' IEEE Transactions on Signal Processing , vol. 58, No. 3, pp. 1847-1862, March 2010.
- [26] Y. Chen, Y. Gu, and A.O. Hero, 'Sparse LMS for system identification,' in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing , pp. 3125-3128, Taipei, May 2009.
- [27] Y. Gu, J. Jin, and S. Mei, ' ℓ 0 norm constraint lms algorithm for sparse system identification,' IEEE Signal Processing Letters , vol. 16, no. 9, pp. 774-777, 2009.
- [28] D. Angelosante, J.A. Bazerque, and G.B. Giannakis, 'Online adaptive estimation of sparse signals: where RLS meets the ℓ 1 -norm,' IEEE Trans. on Signal Processing , vol. 58, no. 7, pp. 3436-3447, July, 2010.
- [29] B. Babadi, N. Kalouptsidis, and V. Tarokh, 'SPARLS: The sparse RLS algorithm,' IEEE Transactions on Signal Processing , vol. 58, no. 8, pp. 4013-4025, Aug., 2010.
- [30] Y. Kopsinis, K. Slavakis, and S. Theodoridis, 'Online sparse system identification and signal reconstruction using projections onto weighted ℓ 1 balls,' IEEE Transactions on Signal Processing , vol. 59, no. 3, pp. 936-952, March, 2010.
- [31] K. Slavakis, Y. Kopsinis, S. Theodoridis, and S. McLaughlin, 'Generalized thresholding and online sparsity-aware learning in a union of

subspaces,' IEEE Transactions on Signal Processing , vol. 61, no. 15, pp. 3760-3773, 2013.

- [32] J. Mota, J. Xavier, P. Aguiar, and M. Puschel, 'Distributed basis pursuit,' IEEE Transactions on Signal Processing , vol. 60, no. 4, April, 2012.
- [33] S. Chouvardas, K. Slavakis, Y. Kopsinis, S. Theodoridis, 'A sparsitypromoting adaptive algorithm for distributed learning,' IEEE Transactions on Signal Processing , vol. 60, no. 10, pp. 5412-5425, Oct. 2012.
- [34] Y. Liu, C. Li and Z. Zhang, 'Diffusion sparse least-mean squares over networks,' IEEE Transactions on Signal Processing , vol. 60, no. 8, pp. 4480-4485, Aug. 2012.
- [35] P. Di Lorenzo, S. Barbarossa, and A. H. Sayed, 'Sparse diffusion LMS for distributed adaptive estimation,' Proc. IEEE Int. Conf. on Acoustics, Speech, and Signal Proc. , pp. 3281-3284, Kyoto, Japan, March 2012.
- [36] P. Di Lorenzo and A. H. Sayed, 'Sparse distributed learning based on diffusion adaptation,' IEEE Transactions on Signal Processing , vol. 61, no. 6, pp. 1419-1433, 15 March 2013.
- [37] S. Sardellitti and S. Barbarossa, 'Distributed RLS estimation for cooperative sensing in small cell networks,' IEEE Inter. Conf. on Acoustics, Speech and Signal Process. , Vancouver, Canada, 2013.
- [38] Z. Liu, Y. Liu and C. Li, 'Distributed sparse recursive least squares over networks,' IEEE Transactions on Signal Processing , vol. 62, no. 6, pp. 1386-1395, March 2014.
- [39] S. L. Lauritzen, Graphical Models , Oxford University Press, 1996.
- [40] A. Wiesel and A. O. Hero III, 'Distributed Covariance Estimation in Gaussian Graphical Models,'IEEE Transactions On Signal Processing, Vol. 60, No. 1, January 2012
- [41] N. Meinshausen and P. Buhlmann, 'High-dimensional graphs and variable selection with the Lasso,' Annals of Statistics , vol. 34, no. 3, pp. 1436-1462, 2006.
- [42] A. Dogandzic and K. Liu, 'Decentralized random-field estimation for sensor networks using quantized spatially correlated data and fusioncenter feedback,' IEEE Trans. Signal Process. , vol. 56, no. 12, pp. 60696085, Dec. 2008.
- [43] E. B. Sudderth, M. J. Wainwright, and A. S. Willsky, 'Embedded trees: Estimation of Gaussian processes on graphs with cycles,' IEEE Transactions on Signal Processing, vol. 52, no. 11, pp. 3136-3150, Nov. 2004.
- [44] V. Delouille, R. N. Neelamani, and R. G. Baraniuk, 'Robust distributed estimation using the embedded subgraphs algorithm,' IEEE Trans. Signal Process. , vol. 54, no. 8, pp. 2998-3010, Aug. 2006.
- [45] V. Chandrasekaran, J. K. Johnson, and A. S. Willsky, 'Estimation in Gaussian graphical models using tractable subgraphs: A walk-sum analysis,' IEEE Trans. Signal Process., vol. 56, no. 5, pp. 1916-1930, May 2008.
- [46] J. Fang and H. Li, ' Distributed estimation of Gauss-Markov random fields with one-bit quantized data,' IEEE Signal Processing Letters , Vol. 17, no. 5, pp. 449-452, May 2010.
- [47] A. Anandkumar, L. Tong, A. Swami, 'Detection of Gauss-Markov Random Fields With Nearest-Neighbor Dependency,' IEEE Trans. on Information Theory , Vol. 55, pp. 816-827, Feb. 2009.
- [48] S. M. Kay, Fundamentals of Statistical Signal Processing: Estimation Theory , Prentice Hall, 1993.
- [49] D. P. Bertsekas and J. N. Tsitsiklis, Parallel and distributed computation: numerical methods , Belmont, MA, Athena Scientific, 1997.
- [50] M. Nevelson and R. Hasminskii, Stochastic approximation and recursive estimation , Providence, Rhode Island: American Math. Soc., 1973.
- [51] E.J. Candes, M. Wakin, and S. Boyd, 'Enhancing sparsity by reweighted ℓ 1 minimization,' Journal of Fourier Analysis and Applications , vol. 14, pp.877-905, 2007.
- [52] M. Yuan, Y. Lin, 'On the non-negative garrotte estimator,' J. Royal Statist. Soc. , vol. 69, pp. 143161, 2007.
- [53] P. Di Lorenzo and S. Barbarossa, 'A bio-inspired swarming algorithm for decentralized access in cognitive radio,' IEEE Trans. on Signal Processing , vol. 59, no. 12, pp. 6160-6174, December 2011.
- [54] H. J. Kushner and G. G. Yin, Stochastic Approximation Algorithms and Applications , New York: Springer-Verlag, 1997.
- [55] A. H. Sayed, Adaptive Filters , Wiley, NJ, 2008.
- [56] R. L. Graham, D. E. Knuth and O. Patashnik, Concrete Mathematics: A Foundation for Computer Science , 2nd ed., Addison-Wesley, 1994.
- [57] T. Kailath, A. H. Sayed, and B. Hassibi, Linear Estimation , Englewood Cliffs, NJ: Prentice-Hall, 2000.
- [58] R. Horn and C. Johnson, Matrix Analysis , Cambridge university press, 2005.