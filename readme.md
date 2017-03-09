## installl

	pip install -t ~/.local/lib/python2.7/site-packages/ dicom 
## try
* preprocess
	* cut
	* ZCA
	* data_aug
* network


## preprocess
预处理方法主要参考[Full Preprocessing Tutorial](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook)

在[comment](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/comments)中，我看到好几处大家提出来的可能的错误
#### problem1
After get_hu_pixels, there are still some participants (e.g. 10f03fe0b77d39c758d6fb12304edfdb) with -2048 values. It seems to me that this is due to this line of code:

	image[image == -2000] = 0
Thanks for the great tutorial!

#### solver
I fixed the issue by changing the above line to this one.

	outside_image = image.min()
	image[image == outside_image] = 0

#### problem2
联通区出错

#### problem3

however it seems that some scans in the full data set are missing the 'SliceLocation' attribute. For example id '08acb3440eb23385724d006403feb585'.

Instead I believe the same information can be retrieved from 'ImagePositionPatient'

In total there are 14 patients impacted by having no SliceLocation - IDs below

'08acb3440eb23385724d006403feb585', '1344e604281e68ac71f25c02e7637992', '1b7ca8dad5c36feb0a6abf8079173e22', '1c05dea5723caf192af34ceb5804468f', '3187b6cf14ed2cbe611b01da5e161f21', '4bf6fb574a2ca2da551609e98a573e54', '70671fa94231eb377e8ac7cba4650dfb', '7bc437435c5677d361177adb80547bd0', '809ae218d8b4a973d11358264e4a0823', '995fc0581ed0e3ba0f97dbd7fe63db59', 'a4fa7dd73e3017b97ac6674a10fac216', 'bbf7a3e138f9353414f2d51f0c363561', 'efd5b9e8cb651e18eba6a9623e36e7be', 'f52bd66210db45189027991781e1162b'

#### solver
无
#### 噪声
3D滤波