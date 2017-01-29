require 'image' 
desiredHeight = 198
desiredWidth= 198
scale = 2 --by how much you data is to be augmented

function crop(x, offsets, width, height)
   height = height or width
   return image.crop(x, offsets[1], offsets[2], offsets[1] + width, offsets[2] + height)
end

 function horizontal_reflection(x)
   return image.hflip(x)
end

 function zoomout(x)
   return image.scale(x, desiredWidth, desiredHeight, 'bilinear')
end


 
function generate_crop_posRan()
  for i =1,2 do    
     w = math.random(1,29)
     h = math.random(1,29)
	 table.insert(CROP_POSrand, {w, h})
  end    
   
end

function data_augmentation(x)
    CROP_POSrand = {}
    generate_crop_posRan()     --2 kinds of crops
	
    if x:dim() == 3 then
      -- jitter for training
       new_x = torch.Tensor(scale,x:size(1),    --number of twists and turns
				  desiredWidth, desiredHeight)
       images = {}
        for i = 1, x:size(1) do
           src = x[i]
           im1 = crop(src, CROP_POSrand[1], desiredWidth, desiredHeight)
			  new_x[1][i]=im1
           table.insert(images,im1 )
           --print(CROP_POSrand[1] )    
         end
	        for i = 1, x:size(1) do
	           src = x[i]
	           im2 = zoomout(crop(src, CROP_POSrand[2], desiredHeight))
	           table.insert(images, horizontal_reflection(im2))
				  new_x[2][i]=horizontal_reflection(im2)
	        end   
        
         
          collectgarbage()
         
    end
    
    return new_x
end
