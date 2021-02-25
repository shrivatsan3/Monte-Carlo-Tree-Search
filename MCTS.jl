using DMUStudent.HW3: DenseGridWorld, visualize_tree 
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate
using POMDPSimulators: RolloutSimulator
using POMDPPolicies: FunctionPolicy
using POMDPModelTools: render
using D3Trees: inchrome
using StaticArrays: SA

############
# Question 4
############

m = DenseGridWorld()

S = statetype(m)
A = actiontype(m)

gamma = discount(m) # Get the discount factor for current model
act = actions(m) # Get the actions available for current model
c = 2
#global sp = SA[0,0]
#global r = 0.0

roll = RolloutSimulator(max_steps=10)
p = FunctionPolicy(s->:right)


# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

#=

*Monte Carlo Tree search*
*Heuristic search based on expansion in the direction of highest reward
*Each iteration adds a new state node 
*n iterations -> n-1 nodes
*Algorithm consists of 4 steps-
    Search - Chose the state action pair with highest UCB 
    Expand - Add a state node and list all actions 
    Rollout - From the leaf state node, do a Monte Carlo simulation which would be the value estimate 
            of the leaf node
    Backup - send the value of leaf node up the heirachy

*Play around with c, MC simulation parameters
=#


function MCTS!(s, act)
    if !haskey(n, (s, first(act)))  # List all actions of Leaf node
        for a in act
            n[(s,a)] = 0
            q[(s,a)] = 0.0
        end
         
        if s == start # First iteration, simply list the actions from start node
            return
        else
            return simulate(roll,m,p,s) # Rollout from leaf node
        end
              
    else
        (~,a) = search(s, act)       # Select the best state action pair based on UCB
        n[(s,a)] += 1 # Increment number of times the state action pair was taken
        (sp,r) = @gen(:sp,:r)(m,s,a)    # Generate the next state
        
        if isterminal(m,sp) # Terminal state is defined [-1,-1], so will give error if not for this IF
            return r
        end

        if !haskey(n, (s, a,sp)) #Count the (s,a,sp) triplet for plotting 
            t[(s,a,sp)] = 0
        else
            t[(s,a,sp)]+=1
        end
  
        qcap = r + gamma*MCTS!(sp, act) #Backup step
        q[(s,a)] += (qcap-q[(s,a)])/n[(s,a)] #Update q(s,a) expectation
        return qcap # Send rollout value up the tree
    end
end


exploration(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa) # exploration term for calculating UCB-1


function search(s, act)

    Ns = sum(n[(s,a)] for a in act)
    getmax = Dict()
    for a in act
        getmax[(s,a)] = q[(s,a)] + c*exploration(n[(s,a)], Ns)
    end
    @show return argmax(getmax) # return action with highest UCB 
    
end

global start = SA[19,19]
go  = time_ns()
for k in 1:8
    MCTS!(start,act)
end
#_argmax(a->q[(s,a)], act)
@show time_ns()-go
inchrome(visualize_tree(q, n, t, SA[19,19]))

