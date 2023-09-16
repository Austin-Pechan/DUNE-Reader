All packages besides tesseract can be downloaded by pip installing. 
For tesseract download refer to the YouTube videos: 
  * https://youtu.be/DG5D8A3zi4o?si=ahb_7F2XOY3zBEQJ  (Windows)
  * https://sammybams.hashnode.dev/how-to-install-tesseract-ocr-on-macos (MAC)

Note 1: The left, right, up, and down parameters are arbitrary for now, but when we get the fixed orientation that the robot camera will take we can get actual preset parameters.

Note 2: If the chips are damaged like the one in "One_ASIC_Image" the code doesn't work well. I tried to do some work with bilateral filtering, but was ultimately unsuccessful.
So, it is important to have relatively clean chips and a clear robot camera. (8 of the chips in "COLDATA" I tested worked very well and I suspect they are clean enough)
