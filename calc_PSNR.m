clear all;
PSNR_bicubic_RGB = zeros(1,5);
PSNR_ESPCN_RGB = zeros(1,5);
PSNR_bicubic_Y = zeros(1,5);
PSNR_ESPCN_Y = zeros(1,5);

for i = 0:4
    filename = strcat('./Set5/original_',int2str(i),'.png');
    original = imread(filename);
    filename = strcat('./Set5/bicubic_',int2str(i),'.png');
    bicubic = imread(filename);
    filename = strcat('./Set5/HR_Y_',int2str(i),'.png');
    HR = imread(filename);
    
    %bicubic2 = imresize(original,1/3);
    %bicubic2 = imresize(bicubic2,3);
    %PSNR_bicubic1 = psnr(original, bicubic);
    %PSNR_bicubic2 = psnr(original, bicubic);
    %PSNR_ESPCN = psnr(original, HR);
    %disp([PSNR_bicubic1, PSNR_bicubic2, PSNR_ESPCN]);
    
    PSNR_bicubic = psnr(original, bicubic);
    PSNR_ESPCN = psnr(original, HR);
    PSNR_bicubic_RGB(i+1) = PSNR_bicubic; 
    PSNR_ESPCN_RGB(i+1) = PSNR_ESPCN; 
    disp('PSNR in RGB')
    disp([PSNR_bicubic, PSNR_ESPCN]);
    
    original = rgb2ycbcr(original);
    bicubic = rgb2ycbcr(bicubic);
    HR = rgb2ycbcr(HR);
    PSNR_bicubic = psnr(original(:,:,1), bicubic(:,:,1));
    PSNR_ESPCN = psnr(original(:,:,1), HR(:,:,1));
    PSNR_bicubic_Y(i+1) = PSNR_bicubic; 
    PSNR_ESPCN_Y(i+1) = PSNR_ESPCN; 
    disp('PSNR in Y')
    disp([PSNR_bicubic, PSNR_ESPCN]);
end

disp('average RGB:')
disp([mean(PSNR_bicubic_RGB), mean(PSNR_ESPCN_RGB)]);

disp('average Y:')
disp([mean(PSNR_bicubic_Y), mean(PSNR_ESPCN_Y)]);
avg_PSNR_bicubic = mean(PSNR_bicubic_Y);
avg_PSNR_ESPCN = mean(PSNR_ESPCN_Y);