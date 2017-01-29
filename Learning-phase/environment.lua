require 'torch'
require 'image'
require 'cunn'
require 'ideAugm'
r_len=8 --recording time in sec
raw_frame_height=320 --- height and width of captured frame
raw_frame_width=240 --- height and width of captured frame
proc_frame_size=198 --
state_size=8
frame_per_sec=1
step=1

function get_data(episode,tsteps)
	local images=torch.Tensor(tsteps,2,state_size,proc_frame_size,proc_frame_size)	
		dirname_rgb='dataset/RGB/ep'..episode
		
		for step=1,tsteps do

			
			local im=torch.Tensor(r_len,raw_frame_width,raw_frame_height)
			for i=1,r_len do
					local filename=dirname_rgb..'/image_'..step..'_'..i..'.png'
			 		local im2 =image.load(filename)
					im[i]=im2[1]
			end
			
			local proc_im=torch.Tensor(2,state_size,proc_frame_size,proc_frame_size)
			proc_im=data_augmentation(im)
			images[step]=proc_im

				collectgarbage()
   			
		end
	return images

end 


