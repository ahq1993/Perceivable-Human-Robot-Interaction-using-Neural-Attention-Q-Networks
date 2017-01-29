require 'DARQN'
require 'TransitionTable'
require 'optim'
require 'environment'
require 'image'
require 'paths'
require 'cunn'

local nql = torch.class('NeuralQLearner')


function nql:__init(args)
    self.state_dim  = 198 -- State dimensionality 84x84.
    self.actions    = {'1','2','3','4'}
    self.n_actions  = #self.actions
    
    --- epsilon annealing
    self.ep_start   = 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = 0.1
    self.ep_endt    = 30000

    ---- learning rate annealing
    self.lr_start       = 0.00025 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = self.lr
    self.lr_endt        = 100000   ---replay memory size
    self.wc             = 0  -- L2 weight cost.
    self.minibatch_size = 25
    self.valid_size     = 500

    --- Q-learning parameters
    self.discount       = 0.99 --Discount factor.
	 
    self.t_steps= args.tsteps
	 
    -- Number of points to replay per learning step.
    self.n_replay       = 1
    -- Number of steps after which learning starts.
    self.learn_start    = 3000
     -- Size of the transition table.
    self.replay_memory  = 30000--10000

    self.hist_len       = 8
    self.clip_delta     = 1
    self.target_q       = 4
    self.bestq          = 0

    self.gpu            = 1

    self.ncols          = 1  -- number of color channels in input
    self.input_dims     = {8, 198, 198}
    self.histType       = "linear"  -- history type to use
    self.histSpacing    = 1
    
    self.bufferSize     =  2000
	
    self.episode=args.epi-1
	collectgarbage()
		local DARQN_args = {
        minibatch_size= self.minibatch_size, input_dims = self.input_dims,
        hist_len = self.hist_len, gpu = self.gpu,
        rnn_size = 256, n_actions = self.n_actions}
	 self.network=DARQN(DARQN_args)
	 if self.target_q  then
		  print ("cloning")
        self.target_network = self.network:clone()
	end
collectgarbage()
	

		  torch.setdefaulttensortype('torch.FloatTensor')
 
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing,bufferSize = self.bufferSize}

    self.transitions =TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
	 self.lastDepth = nil
    self.lastAction = nil
    self.lastTerminal=nil
    self.wc =0
	 self.w, self.dw = self.network:getParameters()
    self.dw:zero()
	 self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)
  
end



function nql:load_data(n1,n2)
	print("loading")
	
	for i=n1,n2 do
			print(i)
			local dirname_rgb='dataset/RGB/ep'..i


			k=0
			for file in paths.iterfiles(dirname_rgb) do
				k=k+1
			end
		   k=k/8
			while k%4 ~=0 do
				k=k-1
			end
			local images=torch.Tensor(k,2,self.hist_len,self.state_dim,self.state_dim)
			images=get_data(i,k)	
			print ("loading done")
			local aset = {'1','2','3','4'}
	
			local rewards=torch.load('files/reward_history.dat')
			local actions=torch.load('files/action_history.dat')
			local ep_rewards=torch.load('files/ep_rewards.dat')
			collectgarbage()
				for step=1,k do
					local terminal =0
					
					if rewards[i][step]>3 then
						self.transitions:add(images[step][1],actions[i][step],1,terminal)
						self.transitions:add(images[step][2],actions[i][step],1,terminal)
					elseif rewards[i][step]<0 then
						self.transitions:add(images[step][1],actions[i][step],-1,terminal)
						self.transitions:add(images[step][2],actions[i][step],-1,terminal)
					else

						self.transitions:add(images[step][1],actions[i][step],0,terminal)
						self.transitions:add(images[step][2],actions[i][step],0,terminal)
					end
		
				end		

			collectgarbage()
	
	end
end 
	
function nql:train()
	
	local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

	local q_max_avg_s={}
	local td_err_s={}
	k=1
	
	for j=1,self.bufferSize, self.minibatch_size do
    local s, a, r, s2, term = self.transitions:sample_y(self.minibatch_size)
		local win=nil
		
			self.dw:zero()
				---here check what is the form of delta,q_all when you input a bactch
			self.network:forward(s)
         self.target_network:forward(s2)
			local deltas = {}
			
			for t=1,self.hist_len do
				local q = self.network.predictions[t]  	
         	local r = r:float()
         	local q2 = self.target_network.predictions[t]
				local q2_max= q2:clone():max(2):mul(self.discount)	
				local target = r:clone()
				q2_max=q2_max:float()
				target:add(q2_max)
				q=q:float()		
				target:resize(self.minibatch_size,1)
				local delta = torch.repeatTensor(target,1,self.n_actions) - q
				a:resize(self.minibatch_size,1)
    			local mask = torch.Tensor():resizeAs(q):fill(0):scatter(2,a,1)
    			delta:cmul(mask)
				if self.clip_delta then
					 delta:clamp(-self.clip_delta, self.clip_delta)
    			end
				deltas[t]=delta:cuda()
				if t==self.hist_len then
						local v_avg=q2_max:float():mean()
						local tderr_avg = deltas[t]
								tderr_avg = tderr_avg:float():abs():mean()
						table.insert(q_max_avg_s,v_avg)
						table.insert(td_err_s,tderr_avg)
						
				end
				
			end
			
			self.network:backward(s, deltas)

    -- add weight cost to gradient
    		self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
			 local t = math.max(0, self.numSteps - self.learn_start)
			 self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
			 self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
			 self.g:mul(0.95):add(0.05, self.dw)
			 self.tmp:cmul(self.dw, self.dw)
			 self.g2:mul(0.95):add(0.05, self.tmp)
			 self.tmp:cmul(self.g, self.g)
			 self.tmp:mul(-1)
			 self.tmp:add(self.g2)
			 self.tmp:add(0.01)
			 self.tmp:sqrt()

			 -- accumulate update
			 self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
			 self.w:add(self.deltas)
			collectgarbage()

  		end
	    collectgarbage()
		return q_max_avg_s,td_err_s 			

end





