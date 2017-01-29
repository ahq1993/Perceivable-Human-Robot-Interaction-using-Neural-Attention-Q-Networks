require 'torch'
require 'environment'
require 'image'
require 'NeuralQLearner'
require 'paths'

local t_steps=2000



local gpu=1

--- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    torch.manualSeed(torch.initialSeed())
    local firstRandInt = torch.random()
    if gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
    end


local win=nil
collectgarbage()
local episode=torch.load('files/episode.dat')
args={epi=episode, tsteps=t_steps}
local agent=NeuralQLearner(args)
collectgarbage()
local win=nil


function main()

local q_max_s_ep=torch.load('files/q_max_s_ep.dat')
local td_s_ep=torch.load('files/td_err_s_ep.dat')


 --- load data
 agent:load_data(1,14)
 --- training 
 collectgarbage()
for j=1,20 do
	local q_s_replay={}
	local t_s={}
 	print("epoch="..j.."/20")
	 for i=1,30 do

		print("iter="..i.."/30")

	 	q_max_avg_s,td_err_s=agent:train()
		table.insert(q_s_replay,q_max_avg_s)
		table.insert(t_s,td_err_s)
		collectgarbage()	
	 end

	agent.target_network=agent.network:clone()

	table.insert(q_max_s_ep,q_s_replay)
 	table.insert(td_s_ep,t_s)
  local model_dir='results/ep'..episode..'-'..j
  paths.mkdir(model_dir)
 	
  local save_model=model_dir..'/model.net'
  torch.save(save_model,agent.network) 	
 
 
end
	

 
 
 local model_dir='results/ep'..episode
 paths.mkdir(model_dir)
 local save_model=model_dir..'/model.net'

 torch.save(save_model,agent.network)
 torch.save('files/q_max_s_ep.dat',q_max_s_ep)
 torch.save('files/td_err_s_ep.dat',td_s_ep)

 collectgarbage()

end
collectgarbage()
main()
