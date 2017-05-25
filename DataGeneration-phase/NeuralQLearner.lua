require 'optim'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'paths'
require 'image'
require 'LinearWithoutBias'

local nql = torch.class('NeuralQLearner')


function nql:__init(args)
    self.state_dim  = 198 -- State dimensionality 84x84.
    self.actions    = {'1','2','3','4'}
    self.n_actions  = #self.actions
    self.win=nil
    --- epsilon annealing
    self.ep_start   = 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = 0.1
    self.ep_endt    = 30000
	 self.learn_start= 4000
     
    self.bufferSize =  3000
	 self.episode=args.epi
	 self.iter=1
    self.seq=""
	
	 self.cnnA=torch.load('results/soft/cnnA.net')
	 self.cnnB=torch.load('results/soft/cnnB.net')
	 self.lstmA=torch.load('results/soft/lstmA.net')
	 self.lstmB=torch.load('results/soft/lstmB.net')
	 self.dqnA=torch.load('results/soft/dqnA.net')
	 self.dqnB=torch.load('results/soft/dqnB.net')
		
	 
    self.prev_c = torch.zeros(1, 256)
    self.prev_h = self.prev_c:clone()

    self.prev_c_2 = torch.zeros(1, 256)
    self.prev_h_2 = self.prev_c_2:clone()
    
    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastDepth = nil
    self.lastAction = nil
    self.lastTerminal=nil
    
	  
end


function nql:predict(rgb)
    local input = rgb:reshape(1, 1, 198, 198):float()
   
    self.prev_c = self.prev_c:float()
    self.prev_h = self.prev_h:float()

    local attention, observation = unpack(self.cnnA:forward{self.prev_h, input})
    local next_c, next_h = unpack(self.lstmA:forward{observation, self.prev_c,self.prev_h})
    self.prev_c:copy(next_c)
    self.prev_h:copy(next_h)
    local prediction = self.dqnA:forward(next_h)

    return attention, prediction   
end


function nql:predict2(dep)
    local input = dep:reshape(1, 1, 198, 198):float()
   
    self.prev_c_2 = self.prev_c_2:float()
    self.prev_h_2 = self.prev_h_2:float()

    local attention, observation = unpack(self.cnnB:forward{self.prev_h_2, input})
   local next_c, next_h = unpack(self.lstmB:forward{observation, self.prev_c_2,self.prev_h_2})
    self.prev_c_2:copy(next_c)
    self.prev_h_2:copy(next_h)
    local prediction = self.dqnB:forward(next_h)

    return attention, prediction       

end

function nql:perceive(reward, state, depth, human, terminal, testing, numSteps, steps, testing_ep) 
  	
    
    local curState = state
    local curDepth = depth  
    local px=-1
    local py=-1
    local actionIndex = 1
    actionIndex, px, py = self:eGreedy(curState,curDepth, numSteps, steps, testing_ep)
    if not terminal then
        return actionIndex, px, py
    else
        return 0,px,py
    end
end


function nql:rng()


	local myString =""
	local myString2 ="1234"
	repeat
		local Choice = math.random(4)
		if string.find(myString,Choice)==nil then
			myString= myString..Choice
		end
	until string.len(myString)==4 
	
	return myString
	
end


function nql:eGreedy(state,depth, numSteps , steps, testing_ep) 
	 self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, numSteps - self.learn_start))/self.ep_endt))
	 ep2=1
    -- Epsilon greedy
	 if torch.uniform() < self.ep then
		  local px=-1
		  local py=-1
		  if steps%4==1 then

				self.seq=self:rng()
				self.iter=1
		  end
				
		  local action= tonumber(string.sub(self.seq,self.iter,self.iter))
		  self.iter=self.iter+1
		  return action,px,py
         else
		  self.iter=self.iter+1
                  return self:greedy(state,depth)
         end
end


function nql:greedy(state,depth) 
	 print("greedy")
         state = state:float()
	 depth=depth:float()
	 local win=nil

	 local upsample = nn.Sequential()
	 upsample:add(nn.SpatialSubSampling(1,9,9,3,3))
	 upsample:add(nn.SpatialSubSampling(1,8,8,2,2))
	 upsample:add(nn.SpatialSubSampling(1,7,7,2,2))
	 upsample:add(nn.SpatialSubSampling(1,6,6,1,1))
	 upsample:float()
	 local w, dw = upsample:getParameters()
	 w:fill(0.25)
	 dw:zero()
	 local empty = torch.zeros(1,198,198):float()
	 upsample:forward(empty)

	 local q_all={}
         local q2_all={}	
	 local Yattention
	 for m=1,8 do

			local attention, q = self:predict(state[m])
			local attention2, q2 = self:predict(depth[m])
			attention = upsample:updateGradInput(empty,attention:float())
			attention = image.scale(attention, 198, 198, 'bilinear')
			Yattention=attention
			q_all[m]=q
			q2_all[m]=q2
			
	 end
	print("Yattention")
	local max=0
	local px=0
	local py=0
	local count=1
	local max=0
	for r=1,198 do
		for e=1,198 do
			if Yattention[1][r][e]>max then
				max=Yattention[1][r][e]
				px=r
				py=e
				count=count+1
			end 
		end
	end
	px=px
	py=py    
        local ts=q_all[8][1][1]+q_all[8][1][2]+q_all[8][1][3]+q_all[8][1][4]
	local td=q2_all[8][1][1]+q2_all[8][1][2]+q2_all[8][1][3]+q2_all[8][1][4]
	local q_fus={}	
	for h=1,4 do
		q_fus[h]=((q_all[8][1][h]/ts)*0.5+(q2_all[8][1][h]/td)*0.5)
	end
   local maxq = q_fus[1]
   local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
   for a = 2, self.n_actions do
        if q_fus[a] > maxq then
            besta = { a }
            maxq = q_fus[a]
        elseif q_fus[a] == maxq then
            besta[#besta+1] = a
        end
   end
   self.bestq = maxq
   local r = torch.random(1, #besta)
   self.lastAction = besta[r]

   return besta[r], px,py
end


