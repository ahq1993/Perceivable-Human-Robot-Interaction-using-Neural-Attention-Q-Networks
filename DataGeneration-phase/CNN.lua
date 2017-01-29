-- adapted from: convnet.lua
gpu=0
noutputs=4
nfeats=1

nstates={16,32,64,256}
filter={9,8,7,6}
stride={3,2,2,1}
poolsize=2

require 'nn'        


local CNN = {}

function CNN.cnn(args)
	

	modelA=nn.Sequential()
	--network:add(nn.Reshape(unpack(args.input_dims)))
	--cov1
	modelA:add(nn.SpatialConvolution(nfeats, nstates[1],filter[1],filter[1],stride[1],stride[1],1))
	modelA:add(nn.ReLU())

	--cov2
	modelA:add(nn.SpatialConvolution(nstates[1],nstates[2],filter[2],filter[2],stride[2],stride[2]))
	modelA:add(nn.ReLU())

	--cov3
	modelA:add(nn.SpatialConvolution(nstates[2],nstates[3],filter[3],filter[3],stride[3],stride[3]))
	modelA:add(nn.ReLU())

	--cov4
	modelA:add(nn.SpatialConvolution(nstates[3],nstates[4],filter[4],filter[4],stride[4],stride[4]))
	modelA:add(nn.ReLU())
	
	    
    local last_num_features = nstates[4]

    -- Add convolutional layers
   


    local n_hiddens = last_num_features

    local x = nn.Identity()()
    local x1 = nn.Identity()(modelA(x))
    local x2 = nn.Reshape(last_num_features, 49)(x1)
    -- batch (32) x features (64) x annotations (49)
    local x3 = nn.SpatialConvolution(last_num_features, n_hiddens, 1, 1, 1, 1)(x1)
    -- batch (32) x hiddens (42) x height (7) x width (7)

    local h = nn.Identity()()
    local h1 = nn.LinearWithoutBias(args.rnn_size, n_hiddens)(h)
			 h1=h1:float()
    -- batch (32) x hiddens (42)
    local h2 = nn.Replicate(49, 3)(h1)
    -- batch (32) x hiddens (42) x annotations (49)
    local h3 = nn.Reshape(n_hiddens, 7, 7)(h2)
    -- batch (32) x hiddens (42) x height (7) x width (7)

    local a1 = nn.Tanh()(nn.CAddTable()({h3, x3}))
    local a2 = nn.SpatialConvolution(n_hiddens, 1, 1, 1, 1, 1)(a1)
    -- batch (32) x softmax (1) x height (7) x width (7)
    local a3 = nn.SoftMax()(nn.Reshape(49)(a2))
    -- batch (32) x annotations (49)
    
    local a4 = nn.Replicate(last_num_features, 2)(a3)
    -- batch (32) x features (64) x annotations (49)

    local context = nn.Sum(3)(nn.CMulTable()({a4, x2}))
    -- batch (32) x features (64)

    local g = nn.gModule({h,x}, {a3,context})

    
		 g:float()


    --if args.verbose >= 100 then
        --graph.dot(g.fg, 'Forward Graph', 'fg')
        --graph.dot(g.bg, 'Backward Graph', 'bg')
    --end

    return g
end


return CNN
