#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkXorImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkThresholdImageFilter.h"
#include "itkScaleTransform.h"
#include "itkImageToVTKImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapToBinaryImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkTestingMacros.h"
#include "itkCropImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageIterator.h"
#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkFlyingEdges3D.h>
#include <vtkSmartPointer.h>
#include <vtkBooleanOperationPolyDataFilter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFillHolesFilter.h>
#include <vtkPCANormalEstimation.h>
#include <vtkSignedDistance.h>
#include <vtkPoissonReconstruction.h>
#include <vtkCleanPolyData.h>
#include <vtkPowerCrustSurfaceReconstruction.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkRenderer.h>
#include <vtkExtractSurface.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkCylinder.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkDistancePolyDataFilter.h>
#include <vtkClipPolyData.h>
#include <vtkImplicitBoolean.h>
#include <vtkStaticPointLocator.h>
#include <vtkSurfaceNets3D.h>
#include <vtkVector.h>
#include <vtkMath.h>
#include <vtkPolyDataNormals.h>
#include <vtkExtractPolyDataGeometry.h>
#include <vtkConvertToPointCloud.h>
#include <vtkPolyDataWriter.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>

#include <vtkPolygon.h> 
#include <vtkPyramid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkImplicitPolyDataDistance.h>
#include "itkEuler3DTransform.h"
#include <vtkDecimatePro.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPLYWriter.h>
#include <vtkPLYReader.h>
#include <cstdlib>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkHausdorffDistancePointSetFilter.h>
#include <vtkDelimitedTextWriter.h>
#include <vtkVariantArray.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkImagingStencilModule.h>
#include <vtkImageStencil.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkMetaImageWriter.h>

#include "C:/Users/DEV/Desktop/RobinStuff/Surface/build/BilateralCannyEdgeDetectionImageFilter.h"


namespace constants {
	const double VOXEL_SIZE = 0.2;
}


// Used to normalize all vectors of a vector image in one pass -> probably a simpler way to do it ?
template <class TInput, class TOutput>
class NormalizeVector
{
public:
	NormalizeVector() = default;
	~NormalizeVector() = default;
	bool
		operator!=(const NormalizeVector&) const
	{
		return false;
	}
	bool
		operator==(const NormalizeVector& other) const
	{
		return !(*this != other);
	}
	inline TOutput
		operator()(const TInput& A) const
	{
		using VectorType = itk::Vector<float, 3>;
		VectorType v;
		v[0] = A[0];
		v[1] = A[1];
		v[2] = A[2];
		v.Normalize();
		TOutput    transformedVector;
		
		transformedVector[0] = v[0];
		transformedVector[1] = v[1];
		transformedVector[2] = v[2];

		return transformedVector;
	}
};

std::array<double,3> imageCoordToPolyDataCoord(std::array<double, 3> voxelPos, itk::Vector<double,3> imageSpacing, itk::Point<double,3> imageOrigin) {
	double coord0 = voxelPos[0] * imageSpacing[0] + imageOrigin[0];
	double coord1 = -(voxelPos[1] * imageSpacing[1] + imageOrigin[1]);
	double coord2 = (voxelPos[2] * imageSpacing[2] + imageOrigin[2]);
	return { coord0 ,coord1 ,coord2 };
}

std::array<double, 3> polyDataCoordToImageCoord(std::array<double, 3> polyDataPos, itk::Vector<double, 3> imageSpacing, itk::Point<double, 3> imageOrigin) {
	double coord0 = (polyDataPos[0] - imageOrigin[0]) / imageSpacing[0];
	double coord1 = (-polyDataPos[1] - imageOrigin[1]) / imageSpacing[1];
	double coord2 = (polyDataPos[2] - imageOrigin[2]) / imageSpacing[2];
	return { coord0 ,coord1 ,coord2 };
}

void GeneratePointCloud(itk::Image<unsigned short, 3>::Pointer image, itk::Image<unsigned short, 3>::SizeType size, itk::Image<unsigned char, 3>::Pointer targetImage) { //Modify 3rd argument to use
	typedef itk::Index<3> indexType;
	vtkNew<vtkPoints> points;
	for (size_t i = 0; i < size[0]; i++) {
		for (size_t j = 0; j < size[1]; j++) {
			for (size_t k = 0; k < size[2]; k++) {
				indexType currIndex;
				currIndex[0] = i;
				currIndex[1] = j;
				currIndex[2] = k;
				if (targetImage->GetPixel(currIndex) != 0.0f) {

					float absXPos = (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0];
					float absYPos = -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]); // Pas oublier le - !!!
					float absZPos = (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2];

					points->InsertNextPoint(absZPos, absYPos, absXPos);
				}
			}
		}
	}

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	vtkNew<vtkPLYWriter> plyWriter;
	plyWriter->SetFileName("Output/PointCloudCanny.ply");
	plyWriter->SetInputData(polyData);
	plyWriter->Write();

	typedef itk::ImageFileWriter<itk::Image<unsigned char, 3>> ImageWriter;
	ImageWriter::Pointer imageWriter = ImageWriter::New();
	imageWriter->SetInput(targetImage);
	imageWriter->SetFileName("Output/PointCloudCanny.mhd");
	imageWriter->Update();
}

void GenerateInitialVolumePointCloud(itk::Image<unsigned short, 3>::Pointer image) {
	typedef itk::Index<3> indexType;
	itk::Image<unsigned short, 3>::SizeType size;
	size = image->GetLargestPossibleRegion().GetSize();

	vtkNew<vtkPoints> points;
	for (size_t i = 0; i < size[0]; i++) {
		for (size_t j = 0; j < size[1]; j++) {
			for (size_t k = 0; k < size[2]; k++) {
				indexType currIndex;
				currIndex[0] = i;
				currIndex[1] = j;
				currIndex[2] = k;
				if (image->GetPixel(currIndex) > 1000.0f) {

					float absXPos = (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0];
					float absYPos = -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]); // Pas oublier le - !!!
					float absZPos = (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2];

					points->InsertNextPoint(absZPos, absYPos, absXPos);
				}
			}
		}
	}

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	vtkNew<vtkPLYWriter> plyWriter;
	plyWriter->SetFileName("C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/Output/PointCloudInitialVolume.ply");
	plyWriter->SetInputData(polyData);
	plyWriter->Write();
}

void PolyDataToImageData(vtkPolyData* polydata) {
	double bounds[6];
	polydata->GetBounds(bounds);
	double spacing[3];
	spacing[0] = constants::VOXEL_SIZE;
	spacing[1] = constants::VOXEL_SIZE;
	spacing[2] = constants::VOXEL_SIZE;
	
	vtkNew<vtkImageData> whiteImage;
	whiteImage->SetSpacing(spacing);
	int dim[3];
	for (int i = 0; i < 3; i++) {
		dim[i] = static_cast<int>(ceil(bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]);
	}
	whiteImage->SetDimensions(dim);
	whiteImage->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);

	double origin[3];
	origin[0] = bounds[0] + spacing[0] / 2;
	origin[1] = bounds[2] + spacing[1] / 2;
	origin[2] = bounds[4] + spacing[2] / 2;
	whiteImage->SetOrigin(origin);
	whiteImage->AllocateScalars(VTK_UNSIGNED_SHORT, 1);
	unsigned char inval = 1600;
	unsigned char outval = 0;
	vtkIdType count = whiteImage->GetNumberOfPoints();
	for (vtkIdType i = 0; i < count; i++) {
		whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
	}

	vtkNew<vtkPolyDataToImageStencil> pol2stenc;
	pol2stenc->SetInputData(polydata);
	pol2stenc->SetOutputOrigin(origin);
	pol2stenc->SetOutputSpacing(spacing);
	pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
	pol2stenc->Update();

	vtkNew<vtkImageStencil> imgstenc;
	imgstenc->SetInputData(whiteImage);
	imgstenc->SetStencilConnection(pol2stenc->GetOutputPort());
	imgstenc->ReverseStencilOff();
	imgstenc->SetBackgroundValue(outval);
	imgstenc->Update();

	vtkNew<vtkMetaImageWriter> writer;
	writer->SetFileName("C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/InitialVolumeFromPolyData.mhd");
	writer->SetInputData(imgstenc->GetOutput());
	writer->Write();
}



int main(int argc, char** argv)
{
	std::string path = "C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/";

	bool writeCanny = true;
	bool writeGradient = true;
	bool boolPadding = false;
	bool cropImage = false;
	bool subVoxRef = true;
	bool gradMagWrite = false;
	bool smoothing = true;
	bool writeOtsu = true;
	bool computePointError = true;
	bool cropPossibleGradientSearchSpace = false;
	int geometry = 0;
	enum recoAlgoEnum { ExtractSurface, Poisson, PowerCrust, SurfReconst, SurfaceNets, FlyingEdges};
	recoAlgoEnum reco = ExtractSurface;
	std::map<recoAlgoEnum, std::string> algoToString = {
		{ExtractSurface, "ExtractSurface"}, {Poisson, "Poisson"}, {PowerCrust, "PowerCrust"}, {SurfReconst, "SurfReconst"}, {SurfaceNets, "SurfaceNets"}, {FlyingEdges, "FlyingEdges"}
	};

    std::string initialMHDFilename = "volRolandHelix";

    	std::cout << "volRolandHelix" << std::endl;
		std::cout.flush();

	if (initialMHDFilename == "Reference")
		cropPossibleGradientSearchSpace = false;


	// Robin parameters
	std::ifstream paramFile(argv[1]);
	std::string line;

	std::getline(paramFile, line);
	writeCanny = stoi(line);

	std::getline(paramFile, line);
	writeGradient = stoi(line);

	std::getline(paramFile, line);
	boolPadding = stoi(line);

	std::getline(paramFile, line);
	cropImage = stoi(line);

	std::getline(paramFile, line);
	subVoxRef = stoi(line);

	std::getline(paramFile, line);
	gradMagWrite = stoi(line);

	std::getline(paramFile, line);
	smoothing = stoi(line);

	std::getline(paramFile, line);
	writeOtsu = stoi(line);

	std::getline(paramFile, line);
	computePointError = stoi(line);

	std::getline(paramFile, line);
	cropPossibleGradientSearchSpace = stoi(line);

	std::getline(paramFile, line);
	geometry = stoi(line);
	std::string geometryString = "";
	if (geometry == 0) { geometryString = "Circular"; }
	else if (geometry == 1) { geometryString = "Helical"; }

	std::getline(paramFile, line);
	reco = static_cast<recoAlgoEnum>(stoi(line));

	std::getline(paramFile, line);
	initialMHDFilename = line;

	std::getline(paramFile, line);
	std::string referenceFilename = line;
	// End of Robin parameters


	// Distance algorithm verification

	/*vtkNew<vtkPoints> points;
	points->InsertNextPoint(0.0, 0.0, 0.0);
	points->InsertNextPoint(2.0, 0.0, 0.0);
	points->InsertNextPoint(2.0, 2.0, 0.0);
	points->InsertNextPoint(0.0, 2.0, 0.0);

	vtkNew<vtkPolygon> polygon;
	polygon->GetPointIds()->SetNumberOfIds(4);
	polygon->GetPointIds()->SetId(0, 0);
	polygon->GetPointIds()->SetId(1, 1);
	polygon->GetPointIds()->SetId(2, 2);
	polygon->GetPointIds()->SetId(3, 3);

	vtkNew<vtkCellArray> polygons;
	polygons->InsertNextCell(polygon);

	vtkNew<vtkPolyData> polygonPolyData;
	polygonPolyData->SetPoints(points);
	polygonPolyData->SetPolys(polygons);

	vtkNew<vtkPoints> pyramidPoints;
	pyramidPoints->InsertNextPoint(0.0, 0.0, 0.0);
	pyramidPoints->InsertNextPoint(2.0, 0.0, 0.0);
	pyramidPoints->InsertNextPoint(2.0, 2.0, 0.0);
	pyramidPoints->InsertNextPoint(0.0, 2.0, 0.0);
	pyramidPoints->InsertNextPoint(1.0, 1.0, 1.0);

	vtkNew<vtkPyramid> pyramid;
	pyramid->GetPointIds()->SetId(0, 0);
	pyramid->GetPointIds()->SetId(1, 1);
	pyramid->GetPointIds()->SetId(2, 2);
	pyramid->GetPointIds()->SetId(3, 3);
	pyramid->GetPointIds()->SetId(4, 4);

	vtkNew<vtkCellArray> pyramids;
	pyramids->InsertNextCell(pyramid);

	vtkNew<vtkPolyData> pyramidPolyData;
	pyramidPolyData->SetPoints(pyramidPoints);
	pyramidPolyData->SetPolys(pyramids);

	vtkNew<vtkUnstructuredGrid> ug;
	ug->SetPoints(pyramidPoints);
	ug->InsertNextCell(pyramid->GetCellType(), pyramid->GetPointIds());
	auto geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
	geometryFilter->SetInputData(ug);
	geometryFilter->Update();


	vtkNew<vtkCleanPolyData> polygonCleanPolyData;
	polygonCleanPolyData->SetInputData(polygonPolyData);

	vtkNew<vtkCleanPolyData> pyramidCleanPolyData;
	pyramidCleanPolyData->SetInputData(geometryFilter->GetOutput());


	vtkSmartPointer<vtkImplicitPolyDataDistance> implicitPolyDataDistance = vtkSmartPointer<vtkImplicitPolyDataDistance>::New();
	implicitPolyDataDistance->SetInput(polygonPolyData);
	double pt0[3] = { 3.0, 1.0, 1.0 };
	std::cout << "Distance : " << implicitPolyDataDistance->EvaluateFunction(pt0) << std::endl;


	vtkNew<vtkDistancePolyDataFilter> distanceTestFilter;
	distanceTestFilter->SetInputConnection(1, polygonCleanPolyData->GetOutputPort());
	distanceTestFilter->SetInputConnection(0, pyramidCleanPolyData->GetOutputPort());
	distanceTestFilter->SetSignedDistance(false);
	distanceTestFilter->Update();

	vtkNew<vtkSTLWriter> stlDistanceWriter;
	stlDistanceWriter->SetFileName(path + "Output/distanceTest.stl");
	stlDistanceWriter->SetInputData(distanceTestFilter->GetOutput());
	stlDistanceWriter->Write();*/


	// End of distance algorithm verification


	typedef unsigned short InputPixelType; // Raw values are encoded in unsigned short
	typedef itk::Image< InputPixelType, 3 > InputImageType;

	typedef unsigned char MaskPixelType;
	typedef itk::Image< MaskPixelType, 3 > MaskImageType;

	typedef float CannyPixelType;
	typedef itk::Image< CannyPixelType, 3 > CannyOutputImageType;

	// Tests Robin
	std::cout << "breakpoint 1" << std::endl;
	std::cout.flush();

	InputImageType::Pointer image;
	try {
		image = itk::ReadImage<InputImageType>("C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/Input/" + initialMHDFilename + ".mhd");
		//image = itk::ReadImage<InputImageType>("C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/Input/volRolandHelix.mhd");
		std::cout << argv[1] << std::endl;
		std::cout.flush();

		// dim tests
		using TestRescaleFilterType = itk::RescaleIntensityImageFilter<InputImageType, MaskImageType>;
		auto dimTest = TestRescaleFilterType::New();
		using ImageCalculatorFilterType = itk::MinimumMaximumImageCalculator<InputImageType>;
		auto imageCalculatorFilter = ImageCalculatorFilterType::New();
		imageCalculatorFilter->SetImage(image);
		imageCalculatorFilter->Compute();
		std::cout << imageCalculatorFilter->GetMinimum() << " " << imageCalculatorFilter->GetMaximum() << std::endl;


		dimTest->SetInput(image);
		dimTest->Update();
		std::cout << dimTest->GetInputMinimum() << " " << dimTest->GetInputMaximum() << " " << sizeof(MaskPixelType) << " " << std::endl;
		// End dim tests

		// Test volume as ply
		//GenerateInitialVolumePointCloud(image);
		// End test volume as ply

	}
	catch (itk::ExceptionObject& ex) {
		std::cout << ex.what() << std::endl;
	}
	std::cout << "breakpoint 2" << std::endl;
	std::cout.flush();
	// Fin tests Robin

	InputImageType::RegionType region = image->GetLargestPossibleRegion();
	InputImageType::SizeType size = region.GetSize();

    // Will pad the image with 0 to avoid border effect when computing the gradient (only needed for mhd generated from the VTK filter)
    // Should be moved elsewhere
	if (boolPadding) {
		using PaddingFilterType = itk::ConstantPadImageFilter<InputImageType, InputImageType>;
		auto padding = PaddingFilterType::New();
		padding->SetInput(image);
		InputImageType::SizeType lowerExtendRegion;
		lowerExtendRegion.Fill(1);
		padding->SetPadLowerBound(lowerExtendRegion);
		padding->SetPadUpperBound(lowerExtendRegion);
		padding->SetConstant(0);
		padding->Update();

		typedef itk::ImageFileWriter<InputImageType> PaddedImageWriter;
		PaddedImageWriter::Pointer paddedImageWriter = PaddedImageWriter::New();
		paddedImageWriter->SetInput(padding->GetOutput());
		paddedImageWriter->SetFileName("PaddedInput.mhd");
		paddedImageWriter->Update();
		
		return true;
	}

	if (cropImage) {
		using CropImageFilterType = itk::CropImageFilter<InputImageType, InputImageType>;
		InputImageType::SizeType cropUp = {50,50,50};
		InputImageType::SizeType cropDown = { 50,0,50 };
		auto cropFilter = CropImageFilterType::New();
		cropFilter->SetInput(image);
		cropFilter->SetUpperBoundaryCropSize(cropUp);
		cropFilter->SetLowerBoundaryCropSize(cropDown);

		typedef itk::ImageFileWriter<InputImageType> CroppedImageWriter;
		CroppedImageWriter::Pointer croppedImageWriter = CroppedImageWriter::New();
		croppedImageWriter->SetInput(cropFilter->GetOutput());
		croppedImageWriter->SetFileName("CroppedInput.mhd");
		croppedImageWriter->Update();
		return true;
	}

	std::cout << size << std::endl;


	// Test itk transform Robin

	/*using ScalarType = float;
	using EulerTransformType = itk::Euler3DTransform<ScalarType>;
	auto eulerTransform = EulerTransformType::New();
	EulerTransformType::ParametersType parameters(6);
	parameters[0] = 1.57;
	parameters[1] = 0;
	parameters[2] = 0;
	parameters[3] = 0;
	parameters[4] = 0;
	parameters[5] = 0;
	eulerTransform->SetParameters(parameters);

	auto resample = itk::ResampleImageFilter<InputImageType, InputImageType>::New();
	resample->SetInput(image);
	resample->SetSize(size);
	resample->SetTransform(eulerTransform);*/


	// fin test itk transform Robin


	//ThresholdImage
	//marchepo en gpu ?
	typedef itk::OtsuThresholdImageFilter<InputImageType, MaskImageType> FilterType; // ushort to char
	FilterType::Pointer filter = FilterType::New();

	filter->SetInput(image);
	filter->SetInsideValue(0);
	filter->SetOutsideValue(1);
	filter->Update();


	// Test Otsu Point cloud
	//std::cout << "size " << size << std::endl;
	//GeneratePointCloud(image, size, filter->GetOutput());
	// end test otsu point cloud

	MaskImageType::Pointer maskImage = filter->GetOutput();

	
	typedef itk::MaskImageFilter<InputImageType, MaskImageType, InputImageType> MaskFilterType;
	auto maskFilter = MaskFilterType::New();
	maskFilter->SetInput(image);
	maskFilter->SetMaskImage(maskImage);
	maskFilter->Update();
	std::cout << "Otsu done !" << std::endl;

	if (writeOtsu) {
		typedef itk::ImageFileWriter<InputImageType> ImageWriter;
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(maskFilter->GetOutput());
		imageWriter->SetFileName("TestOtsu.mhd");
		imageWriter->Update();
		std::cout << "Otsu write done !" << std::endl;
	}

	/*
	typedef itk::ThresholdImageFilter< InputImageType > ThresholdFilterType;
	auto thresholdFilter = ThresholdFilterType::New();

	thresholdFilter->SetOutsideValue(0);
	thresholdFilter->SetInput(maskFilter->GetOutput());
	thresholdFilter->ThresholdBelow(15000);
	thresholdFilter->Update();
	*/
	using CastFilterType = itk::CastImageFilter<InputImageType, CannyOutputImageType>;
	auto castFilter = CastFilterType::New();
	castFilter->SetInput(maskFilter->GetOutput()); // Takes InputImageType(ushort) as input and casts them into char but why and why 0 - 65535 instead of 0-255 ?


	// Test bilateral filter
	/*typedef itk::BilateralCannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> BilateralCannyFilter;
	BilateralCannyFilter::Pointer bilateralCanny = BilateralCannyFilter::New();
	bilateralCanny->SetInput(castFilter->GetOutput());
	bilateralCanny->SetDomainSigmas(0.3);
	bilateralCanny->SetRangeSigma(2000);
	if (initialMHDFilename == "Reference") {
		bilateralCanny->SetLowerThreshold(0.1f);
		bilateralCanny->SetUpperThreshold(0.9f);
	}
	else {
		bilateralCanny->SetLowerThreshold(2400.f); // 800
		bilateralCanny->SetUpperThreshold(2500.f); // 2500
	}*/


	
	typedef itk::CannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> CannyFilter;
	CannyFilter::Pointer canny = CannyFilter::New();
	canny->SetInput(castFilter->GetOutput());
	if (initialMHDFilename == "Reference") {
		canny->SetLowerThreshold(0.1f);
		canny->SetUpperThreshold(0.9f);
	}
	else {
		canny->SetLowerThreshold(800.f); // 800
		canny->SetUpperThreshold(2500.f); // 2500
	}
	
	std::cout << "Threshold : " << canny->GetLowerThreshold() << " " << canny->GetUpperThreshold() << " " << canny->GetMaximumError() << std::endl;
	canny->SetVariance(0.1);

	using RescaleFilterType = itk::RescaleIntensityImageFilter<CannyOutputImageType, MaskImageType>;
	auto rescale = RescaleFilterType::New();



	// TODO sufit d'aller chercher la gradient de base enft ... juste il me faut � la fois la direction et � la fois la norme non ?
	// faire par r�gion pour optimiser la m�moire !!!
	
	//rescale->SetInput(bilateralCanny->GetOutput()); // Robin Canny
	rescale->SetInput(canny->GetOutput()); // Roland Canny
	//rescale->SetInput(castFilter->GetOutput());
	rescale->Update();

	std::cout << rescale->GetInputMinimum() << " " << rescale->GetInputMaximum() << std::endl;
	
	MaskImageType::Pointer cannyImage = rescale->GetOutput();
	std::cout << "Canny done !" << std::endl;

	GeneratePointCloud(image, size, cannyImage); // To delete
	
	typedef itk::Index<3> indexType;
	using DuplicatorType = itk::ImageDuplicator<MaskImageType>;
	auto duplicator = DuplicatorType::New();
	duplicator->SetInputImage(cannyImage);
	duplicator->Update();

	MaskImageType::Pointer cannyImageBordered = duplicator->GetOutput();

	// Construi un voisinage autour des points de Canny pour l'image du gradient (peut faire une opération morphologique à la place)
	if (cropPossibleGradientSearchSpace){
		for (size_t i = 0; i < size[0]; i++) {
			for (size_t j = 0; j < size[1]; j++) {
				for (size_t k = 0; k < size[2]; k++) {
					indexType currIndex;
					indexType index;
					currIndex[0] = i;
					currIndex[1] = j;
					currIndex[2] = k;
					if (cannyImage->GetPixel(currIndex) != 0.0f) {
						for (int l = -2; l < 3; l++){
							index[0] = currIndex[0] + l;
							if (((int)currIndex[0] + l < 0)|| ((int)currIndex[0] + l > size[0])) {
								continue;
							}
							for (int m = -2; m < 3; m++) {
								if (((int)currIndex[1] + m < 0) || ((int)currIndex[1] + m > size[1])) {
									continue;
								}
								index[1] = currIndex[1] + m;
								for (int n = -2; n < 3; n++) {
									if (((int)currIndex[2] + n < 0) || ((int)currIndex[2] + n > size[2])) {
										continue;
									}
									index[2] = currIndex[2] + n;
									cannyImageBordered->SetPixel(index, 255);
								}
							}
						}
					}
				}
			}
		}
	}
	if (writeCanny) {

		typedef itk::ImageFileWriter<MaskImageType> ImageWriter;
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(cannyImage);
		imageWriter->SetFileName("C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/TestCanny.mhd");
		imageWriter->Update();
		std::cout << "Canny write done !" << std::endl;
	}
	std::string stlFilename;
	std::string compareFileName;
	std::string errorFilename;
	vtkNew<vtkPolyData> targetPolyData;

	typedef itk::Image<float, 3> GradientImageType;

	typedef itk::GradientMagnitudeImageFilter<InputImageType, GradientImageType> GradientImageFilterType;
	GradientImageFilterType::Pointer gradientMagFilter = GradientImageFilterType::New();
	gradientMagFilter->SetInput(image);
	gradientMagFilter->Update();
	GradientImageType::Pointer gradientMagImage = gradientMagFilter->GetOutput();
	std::cout << "Gradient done !" << std::endl;

	if (writeGradient) { // How to write itk files
		typedef itk::ImageFileWriter<GradientImageType> GradientWriter;
		GradientWriter::Pointer gradientWriter = GradientWriter::New();
		gradientWriter->SetInput(gradientMagImage);
		gradientWriter->SetFileName("TestGrad.mhd");
		gradientWriter->Update();
		std::cout << "Gradient writing done !" << std::endl;
	}

	using GradientFilterType = itk::GradientImageFilter<InputImageType, float>;
	auto gradientFilter = GradientFilterType::New();
	gradientFilter->SetInput(image);
	gradientFilter->Update();
	auto gradientVecImage = gradientFilter->GetOutput();


	if (gradMagWrite) {
		typedef itk::CovariantVector<float, 3> GradMagVector;
		typedef itk::Image< GradMagVector, 3> GradientMagnitudeImageType;
		typedef itk::ImageFileWriter<GradientMagnitudeImageType> GradMagImageWriter;
		GradMagImageWriter::Pointer gradMagImageWriter = GradMagImageWriter::New();
		typedef itk::UnaryFunctorImageFilter<GradientMagnitudeImageType, GradientMagnitudeImageType, NormalizeVector<GradMagVector, GradMagVector> > GradImageNormFilter;
		auto gradImageNormFilter = GradImageNormFilter::New();
		gradImageNormFilter->SetInput(gradientVecImage);
		auto gradMagImageNormalized = gradImageNormFilter->GetOutput();
		gradMagImageWriter->SetInput(gradientVecImage);
		gradMagImageWriter->SetFileName("TestGradMag.mhd");
		gradMagImageWriter->Update();
		std::cout << "GradImage write done !" << std::endl;
	}

	using MaskFilterGradType = itk::MaskImageFilter<GradientImageType, MaskImageType>;
	auto maskGradFilter = MaskFilterGradType::New();
	if (cropPossibleGradientSearchSpace) {
		maskGradFilter->SetInput(gradientMagImage);
		maskGradFilter->SetMaskImage(cannyImageBordered);
		maskGradFilter->Update();
		typedef itk::ImageFileWriter<GradientImageType> GradientWriter;
		GradientWriter::Pointer gradientWriter = GradientWriter::New();
		gradientWriter->SetInput(maskGradFilter->GetOutput());
		gradientWriter->SetFileName("TestGrad.mhd");
		gradientWriter->Update();
		std::cout << "Gradient writing done !" << std::endl;
	}

	using InitImageBSplineInterp = itk::BSplineInterpolateImageFunction<GradientImageType>;
	auto bSplineInterpFilter = InitImageBSplineInterp::New();
	bSplineInterpFilter->SetSplineOrder(3);
	if (cropPossibleGradientSearchSpace)
		bSplineInterpFilter->SetInputImage(maskGradFilter->GetOutput());
	else
		bSplineInterpFilter->SetInputImage(gradientMagImage);



	if (reco == SurfaceNets || reco == FlyingEdges) {
		vtkNew<vtkSTLWriter> stlWriter;
		vtkPolyData* polyDataNotRotated;
		using FillerType = itk::BinaryFillholeImageFilter<MaskImageType>;
		auto filler = FillerType::New();
		filler->SetInput(cannyImage);
		filler->SetForegroundValue(255);
		auto filledImage = filler->GetOutput();

		// Test Robin
		typedef itk::ImageFileWriter<itk::Image<unsigned short, 3>> ImageWriter;
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(maskFilter->GetOutput());
		imageWriter->SetFileName("Output/initialImageToDelete.mhd");
		imageWriter->Update();



		using ITKToVTK = itk::ImageToVTKImageFilter<MaskImageType>; // Roland
		//using ITKToVTK = itk::ImageToVTKImageFilter<InputImageType>; // Robin
		auto ITKToVTkConverter = ITKToVTK::New();
		ITKToVTkConverter->SetInput(filledImage); //Roland
		//ITKToVTkConverter->SetInput(maskFilter->GetOutput()); //Robin
		ITKToVTkConverter->Update();
		// Faire une fonction imageToTriangulation
		vtkNew<vtkTransform> transP1;
		transP1->Scale(1, -1, 1);

		if (geometry == 0) {
			transP1->RotateY(-90);
		}
		vtkNew<vtkTransformPolyDataFilter> imageDataCorrect;


		// Robin reconstructed volume transform
		if (geometry == 1) {
			transP1->RotateY(90);
		}
			
		// End of Robin volume transform
		
		if (reco == SurfaceNets) {
			/*
			using BinaryImageToLabelMapFilterType = itk::BinaryImageToLabelMapFilter<MaskImageType>;
			auto binaryImageToLabelMapFilter = BinaryImageToLabelMapFilterType::New();
			binaryImageToLabelMapFilter->SetInput(filler->GetOutput());
			binaryImageToLabelMapFilter->Update();
			*/
			vtkNew<vtkSurfaceNets3D> surfaceNets;
			surfaceNets->SetInputData(ITKToVTkConverter->GetOutput());
			surfaceNets->SetValue(0, 255); // Valeurs de Labels
			//surfaceNets->SmoothingOff();
			//surfaceNets->SetOutputMeshTypeToTriangles();
			surfaceNets->Update();
			polyDataNotRotated = surfaceNets->GetOutput();
			stlFilename = path + "Output/SurfaceNets/" + initialMHDFilename + ".stl";
			compareFileName = path + "Output/SurfaceNets/" + initialMHDFilename + "Comp.vtk";
			errorFilename = path + "Output/SurfaceNets/" + initialMHDFilename + "Errors.txt";
			imageDataCorrect->SetInputData(polyDataNotRotated);
			imageDataCorrect->SetTransform(transP1);
			imageDataCorrect->Update();
			/*
			stlWriter->SetFileName(stlFilename.c_str());
			stlWriter->SetInputData(imageDataCorrect->GetOutput());
			stlWriter->Write();
			*/
			targetPolyData->DeepCopy(imageDataCorrect->GetOutput());
			
		} else {
			// No included smoothing mechanism with flyingedges
			vtkNew<vtkFlyingEdges3D> flyingEdges;
			flyingEdges->SetInputData(ITKToVTkConverter->GetOutput());
			flyingEdges->SetValue(0, 255); //Roland
			//flyingEdges->SetValue(0, 30000); //Robin
			flyingEdges->ComputeNormalsOn();
			flyingEdges->ComputeGradientsOn();
			flyingEdges->Update();
			polyDataNotRotated = flyingEdges->GetOutput();
			stlFilename = path + "Output/FlyingEdges/" + initialMHDFilename + ".stl";
			compareFileName = path + "Output/FlyingEdges/" + initialMHDFilename + "Comp.vtk";
			errorFilename = path + "Output/FlyingEdges/" + initialMHDFilename + "Errors.txt";
			imageDataCorrect->SetInputData(polyDataNotRotated);
			imageDataCorrect->SetTransform(transP1);
			imageDataCorrect->Update();
			stlWriter->SetFileName(stlFilename.c_str());
			stlWriter->SetInputData(imageDataCorrect->GetOutput());
			stlWriter->Write();
			targetPolyData->DeepCopy(imageDataCorrect->GetOutput());
		}

		
		if (smoothing) { 
			vtkSmartPointer<vtkWindowedSincPolyDataFilter> smooth = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
			smooth->SetInputData(imageDataCorrect->GetOutput());
			smooth->SetPassBand(0.01);
			//smooth->BoundarySmoothingOff(); // Recheck
			smooth->SetNumberOfIterations(20);
			//smooth->FeatureEdgeSmoothingOff(); // Recheck
			smooth->NonManifoldSmoothingOn();
			smooth->NormalizeCoordinatesOn();
			smooth->Update();
			targetPolyData->DeepCopy(smooth->GetOutput());
		}

		if (subVoxRef) {

			// Retrieve normals either with vtkArrayDownCast<vtkFloatArray>(output->GetPointData()->GetNormals()) or vtkArrayDownCast<vtkFloatArray>(output->GetPointData()->GetArray("Normals"))
			// As in line 913

			// Life hack
			vtkNew<vtkTransform> transP2;
			vtkNew<vtkTransform> transP3;
			if (geometry == 0) {
				transP2->RotateY(90);
				transP3->RotateY(-90);
			}
			if (geometry == 1) {
				transP2->RotateY(-90);
				transP3->RotateY(90);
			}
			
			imageDataCorrect->SetInputData(targetPolyData);
			imageDataCorrect->SetTransform(transP2);
			imageDataCorrect->Update();
			vtkNew<vtkPolyData> lifeHackPart1;
			lifeHackPart1->DeepCopy(imageDataCorrect->GetOutput());

			
			vtkNew<vtkIdList> listOfPoints;
			double currPointCoord[3];
			itk::CovariantVector< float, 3 > currDir;
			itk::CovariantVector< float, 3 > step;
			itk::CovariantVector< float, 3 > brokenDir; // Des fois nan apparait dans la direction du gradient
			double coordToCheck[3] = { 7.199999809265137, -1.2000000476837158, 9.800000190734863 }; // Point on the edge
			brokenDir.Fill(0); // Les dir avec des nan sont set à 0 et ignorés
			vtkNew<vtkPolyDataNormals> normals;
			normals->SetInputData(lifeHackPart1);
			normals->SetComputePointNormals(true);
			normals->SetAutoOrientNormals(true);
			normals->SetFeatureAngle(75.0);
			normals->Update();
			lifeHackPart1->DeepCopy(normals->GetOutput());

			// On pourait mesurer l'erreur à chaque iteration
			for (size_t iter = 0; iter < 6; iter++) {
				std::cout << "Entering loop iter " << iter << std::endl;
				int nbOfPoints = lifeHackPart1->GetNumberOfPoints();
				std::cout << nbOfPoints << std::endl;
				vtkNew<vtkPoints> newPoints;
				int modifiedPoints = 0;
				vtkFloatArray* normalsData = vtkArrayDownCast<vtkFloatArray>(lifeHackPart1->GetPointData()->GetNormals());
				for (int i = 0; i < nbOfPoints; i++) {
					lifeHackPart1->GetPoint(i, currPointCoord);
					std::array<double, 3> imageIndex = polyDataCoordToImageCoord({ currPointCoord[0],currPointCoord[1],currPointCoord[2] }, image->GetSpacing(), image->GetOrigin());
					std::array<int, 3> roundedImageIndex;
					roundedImageIndex[0] = imageIndex[0];
					roundedImageIndex[1] = imageIndex[1];
					roundedImageIndex[2] = imageIndex[2];
					// gardientImageindex will have the correct position
					InputImageType::IndexType gardientImageindex{ {roundedImageIndex[0], roundedImageIndex[1], roundedImageIndex[2]} };
					
					if (iter == 0) {
						double* currTuple = normalsData->GetTuple(i);
						currDir[0] = currTuple[0];
						currDir[1] = currTuple[1];
						currDir[2] = currTuple[2];
					}
					else {
						currDir = gradientVecImage->GetPixel(gardientImageindex);
					}
					//std::cout << currDir[0] << " " << currDir[1] << " " << currDir[2] << std::endl;
					if (currDir == brokenDir) {
						// A la place de skip les points faire de l'interpolation avec les points autour
						newPoints->InsertNextPoint(currPointCoord[0], currPointCoord[1], currPointCoord[2]);
						continue;
					}
					currDir.Normalize();
					step = currDir * 0.05f * (1.0f/(static_cast<float>(iter)*20.f+1.0f)); // The voxel size is defined by the image spacing (the 0.1f is a subvoxel refinement)
					if (i == 20000) {
						float x = 0.05f * (1.0f / (static_cast<float>(iter) * 20.f + 1.0f));
						std::cout << "Step : " << x << std::endl;
						std::cout << "Current Direction : " << currDir << std::endl;
					}
					
					float maxValue = 0;
					int maxIndex = 0;
					bool printInterp = false;
					/*if (i == 8000) { // Debug
						printInterp = true;
						std::cout << "Point ID : " << i << std::endl;
						std::cout << "CurrDir : " << currDir << std::endl;
						std::cout << "Gradient Image Index : " << gardientImageindex << std::endl;
						std::cout << "Step : " << step << std::endl;
						std::cout << "CurrPointCoord : " << currPointCoord[0] << " " << currPointCoord[1] << " " << currPointCoord[2] << std::endl;
					}*/
					for (int j = -20; j < 21; j++) { // Make it a soft variable
						itk::ContinuousIndex<double, 3> interpCoord;
						interpCoord.Fill(0);
						interpCoord[0] = imageIndex[0] + j * step[0];
						interpCoord[1] = imageIndex[1] + j * step[1];
						interpCoord[2] = imageIndex[2] + j * step[2];
						double interpValue = bSplineInterpFilter->EvaluateAtContinuousIndex(interpCoord);
						
						if (interpValue > maxValue) { 
							maxValue = interpValue;
							maxIndex = j;
						}
						if (printInterp) {
							std::cout << j << " : Interp : " << interpValue << std::endl;
						}
							
					}
					std::array<double, 3> newCoords = imageCoordToPolyDataCoord({ imageIndex[0] + maxIndex * step[0],imageIndex[1] + maxIndex * step[1],imageIndex[2] + maxIndex * step[2] }, image->GetSpacing(), image->GetOrigin());
					
					if (printInterp) {
						std::cout << "Max index : " << maxIndex << std::endl;
					}
					
					newPoints->InsertNextPoint(newCoords[0], newCoords[1], newCoords[2]);
					if (maxIndex != 0)
						modifiedPoints++;
				}
				std::cout << "subVoxelRefinment done Mod points : " << modifiedPoints << std::endl;
				lifeHackPart1->SetPoints(newPoints);
			}

			imageDataCorrect->SetInputData(lifeHackPart1);
			imageDataCorrect->SetTransform(transP3);
			imageDataCorrect->Update();
			targetPolyData->DeepCopy(imageDataCorrect->GetOutput());
		}
		

		stlWriter->SetFileName(stlFilename.c_str());
		stlWriter->SetInputData(targetPolyData);
		stlWriter->Write();

	}  else { // Switching methods

		/*typedef itk::Index<3> indexType;
		// On ne fait plus le centre de gravité on fait directement le rafinment sous voxelique sur les centre des points de Canny
		vtkNew<vtkPoints> points;
		for (size_t i = 0; i < size[0]; i++) {
			for (size_t j = 0; j < size[1]; j++) {
				for (size_t k = 0; k < size[2]; k++) {
					indexType currIndex;
					indexType index;
					currIndex[0] = i;
					currIndex[1] = j;
					currIndex[2] = k;
					if (cannyImage->GetPixel(currIndex) != 0.0f) {

						float absXPos = (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0] ;
						float absYPos = -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]); // Pas oublier le - !!!
						float absZPos = (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2];

						points->InsertNextPoint(absZPos, absYPos, absXPos);
						errorFilename = path + "Output/PointsError" + initialMHDFilename +".txt";
						//std::cout << "Position : " << xPos << " " << yPos << " " << zPos << std::endl;
					}
				}
			}
		}*/

		// Generate point cloud based on the center of gravity technique instead of directly using the voxels positions given by Canny
		// Goal is to try different window sizes to find the optimal one.
		typedef itk::Index<3> indexType;
		vtkNew<vtkPoints> points;
		for (size_t i = 0; i < size[0]; i++) {
			for (size_t j = 0; j < size[1]; j++) {
				for (size_t k = 0; k < size[2]; k++) {
					indexType currIndex;
					indexType index;
					currIndex[0] = i;
					currIndex[1] = j;
					currIndex[2] = k;
					if (cannyImage->GetPixel(currIndex) != 0.0f) {

						float absXPos = (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0];
						float absYPos = -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]); // Pas oublier le - !!!
						float absZPos = (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2];

						points->InsertNextPoint(absZPos, absYPos, absXPos);
						errorFilename = path + "Output/PointsError" + initialMHDFilename + ".txt";
						//std::cout << "Position : " << xPos << " " << yPos << " " << zPos << std::endl;
					}
				}
			}




		std::ofstream pointCloud;
		pointCloud.open(path + "Output/PointCloud" + initialMHDFilename + ".txt");
		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);


		// Test ply before subvoxel
		vtkNew<vtkPLYWriter> plyWriter;
		/*plyWriter->SetFileName(path + "Output/PointCloudBeforeSubvoxel.ply");
		plyWriter->SetInputData(polyData);
		plyWriter->Write();*/

		vtkNew<vtkPolyData> centerPolyData; // Obtained from canny, without subvoxel refinment
		centerPolyData->DeepCopy(polyData);

		// End test ply


		double currPointCoord[3];
		itk::CovariantVector< float, 3 > currDir;
		itk::CovariantVector< float, 3 > step;
		itk::CovariantVector< float, 3 > brokenDir; // BrokenDir
		brokenDir.Fill(0);

		if (true) {
			// The same code should be put in a function with the other one
			for (size_t iter = 0; iter < 6; iter++) {
				int nbOfPoints = polyData->GetNumberOfPoints();
				vtkNew<vtkPoints> newPoints;
				int modifiedPoints = 0;
				for (int i = 0; i < nbOfPoints; i++) {
					polyData->GetPoint(i, currPointCoord);
					std::array<double, 3> imageIndex = polyDataCoordToImageCoord({ currPointCoord[2],currPointCoord[1],currPointCoord[0] }, image->GetSpacing(), image->GetOrigin());
					std::array<int, 3> roundedImageIndex;
					roundedImageIndex[0] = imageIndex[0];
					roundedImageIndex[1] = imageIndex[1];
					roundedImageIndex[2] = imageIndex[2];
					// gardientImageindex will have the correct position
					InputImageType::IndexType gardientImageindex{ {roundedImageIndex[0], roundedImageIndex[1], roundedImageIndex[2]} };

					currDir = gradientVecImage->GetPixel(gardientImageindex);
					//std::cout << currDir[0] << " " << currDir[1] << " " << currDir[2] << std::endl;
					if (currDir == brokenDir) {
						//std::cout << "Broken Dir Coord : " << currPointCoord[0] << " " << currPointCoord[1] << " " << currPointCoord[2] << std::endl;
						newPoints->InsertNextPoint(currPointCoord[0], currPointCoord[1], currPointCoord[2]);
						continue;
					}
					currDir.Normalize();
					step = currDir * 0.05f * (1.0f / (static_cast<float>(iter) * 20.f + 1.0f)); // The voxel size is defined by the image spacing (the 0.1f is a subvoxel refinement)
					
					float maxValue = 0;
					int maxIndex = 0;
					bool printInterp = false;
					if (i == 8000) {
						printInterp = true;
						std::cout << "Point ID : " << i << std::endl;
						std::cout << "CurrDir : " << currDir << std::endl;
						std::cout << "Gradient Image Index : " << gardientImageindex << std::endl;
						std::cout << "Step : " << step << std::endl;
						std::cout << "CurrPointCoord : " << currPointCoord[0] << " " << currPointCoord[1] << " " << currPointCoord[2] << std::endl;
					}
					for (int j = -20; j < 21; j++) {
						itk::ContinuousIndex<double, 3> interpCoord;
						interpCoord.Fill(0);
						interpCoord[0] = imageIndex[0] + j * step[0];
						interpCoord[1] = imageIndex[1] + j * step[1];
						interpCoord[2] = imageIndex[2] + j * step[2];
						double interpValue = bSplineInterpFilter->EvaluateAtContinuousIndex(interpCoord);

						if (interpValue > maxValue) {
							maxValue = interpValue;
							maxIndex = j;
						}
						if (printInterp) {
							std::cout << j << " : Interp : " << interpValue << std::endl;
						}

					}
					std::array<double, 3> newCoords = imageCoordToPolyDataCoord({ imageIndex[0] + maxIndex * step[0],imageIndex[1] + maxIndex * step[1],imageIndex[2] + maxIndex * step[2] }, image->GetSpacing(), image->GetOrigin());

					if (printInterp) {
						std::cout << "Max index : " << maxIndex << std::endl;
					}

					newPoints->InsertNextPoint(newCoords[2], newCoords[1], newCoords[0]);
					if (maxIndex != 0)
						modifiedPoints++;
				}
				std::cout << "subVoxelRefinment done Mod points : " << modifiedPoints << std::endl;
				polyData->SetPoints(newPoints);
			}
		}

		for (size_t i = 0; i < polyData->GetNumberOfPoints(); i++){
			polyData->GetPoint(i, currPointCoord);
			pointCloud << currPointCoord[0] << " " << currPointCoord[1] << " " << currPointCoord[2] << "\n";
		}


		// If we want to write cloud point after subvoxel correction
		/*plyWriter->SetFileName(path + "Output/PointCloudAfterSubvoxel.ply");
		plyWriter->SetInputData(polyData);
		plyWriter->Write();*/


		if (computePointError) {
			// Length of edges
			std::ofstream errorFile;
			errorFile.open(errorFilename);
			vtkNew<vtkStaticPointLocator> pointLocator;
			pointLocator->SetDataSet(polyData);
			pointLocator->BuildLocator();

			// Hardcoded array because flemme
			double point1Coord[16][3] = { {-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},
										  {20.0, 20.0, 20.0},{20.0, 20.0, 20.0},{20.0, 20.0, -20.0},
										  {20.0, 20.0, -20.0},{-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},
										  {-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},{20.0, -20.0, 20.0},
										  {0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0} };

			double point2Coord[16][3] = { {20.0, 20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, -20.0},
										  {20.0, 20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
										  {-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
										  {20.0, -20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
										  {-20.0, 20.0, 20.0},{-20.0, 20.0, -20.0},{20.0, 20.0, -20.0},{20.0, 20.0, 20.0} };
			double distancesError[16];
			double mean = 0;
			double min = 5000;
			double max = 0;
			double stddev = 0;
			// Loop through all 16 outer edges
			for (size_t i = 0; i < 16; i++) {
				vtkIdType point1ID = pointLocator->FindClosestPoint(point1Coord[i]);
				double* point1Point = polyData->GetPoint(point1ID);
				double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
				vtkIdType point2ID = pointLocator->FindClosestPoint(point2Coord[i]);
				double* point2Point = polyData->GetPoint(point2ID);
				double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
				double distance = std::sqrt(vtkMath::Distance2BetweenPoints(point1, point2));
				//compute distance btw the two points
				if (i < 12) {
					distancesError[i] = abs(distance - 40.0);
				}
				else {
					distancesError[i] = abs(distance - 41.2309);
				}
				std::cout << distancesError[i] << " ";
				errorFile << distancesError[i] << " ";
				mean += distancesError[i];
				if (distancesError[i] < min)
					min = distancesError[i];
				if (distancesError[i] > max)
					max = distancesError[i];
			}
			mean = mean / 16.0;
			std::cout << std::endl;
			errorFile << "\n";
			for (size_t i = 0; i < 16; ++i) {
				stddev += pow(distancesError[i] - mean, 2);
			}
			errorFile << "Mean : " << mean << "\n";
			errorFile << "Min : " << min << "\n";
			errorFile << "Max : " << max << "\n";
			errorFile << "StdDev : " << stddev << "\n";
			// Angles
			double anglesErrors[3]; // Voir si j'en fais plus ?
			double realAngles[3] = { 90.0, 90.0, 43.313 };
			double point1Vec[4][3] = { {-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},
										  {-20.0, 20.0, 20.0} };
			double point2Vec[4][3] = { {20.0, -20.0, 20.0},{-20.0, -20.0, -20.0},{-20.0, 20.0, 20.0},
										  {0.0, 50.0, 0.0} };
			// Flemme de la boucle polalalalala
			std::vector<vtkVector3d> vectors;
			for (size_t i = 0; i < 4; i++) {
				vtkIdType point1ID = pointLocator->FindClosestPoint(point1Vec[i]);
				double* point1Point = polyData->GetPoint(point1ID);
				double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
				vtkIdType point2ID = pointLocator->FindClosestPoint(point2Vec[i]);
				double* point2Point = polyData->GetPoint(point2ID);
				double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
				double diff[3] = { point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2] };
				//create vector
				vtkVector3d vector = vtkVector3d(diff);
				vectors.push_back(vector);
			}

			anglesErrors[0] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[2].GetData())) - realAngles[0];
			anglesErrors[1] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[1].GetData())) - realAngles[1];
			anglesErrors[2] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[3].GetData(), vectors[2].GetData())) - realAngles[2];

			std::cout << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << std::endl;
			errorFile << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << "\n";
			errorFile.close();

			vtkNew<vtkCylinder> cylinder;
			cylinder->SetCenter(-20.0, 0.0, 0.0);
			cylinder->SetRadius(10.15);
			cylinder->SetAxis(1.0, 0.0, 0.0);
			vtkNew<vtkImplicitBoolean> boolean;
			boolean->AddFunction(cylinder);
			vtkNew<vtkExtractPolyDataGeometry> extractPolyDataGeometry;
			extractPolyDataGeometry->SetInputData(polyData);
			extractPolyDataGeometry->SetExtractInside(true);
			extractPolyDataGeometry->SetImplicitFunction(boolean);
			extractPolyDataGeometry->Update();

			vtkNew<vtkConvertToPointCloud> pcConvert;
			pcConvert->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
			pcConvert->SetCellGenerationMode(vtkConvertToPointCloud::NO_CELLS);
			pcConvert->Update();
			vtkPolyData* cylPoints = pcConvert->GetOutput();
			std::ofstream myfile;
			myfile.open(path + "Output/" + initialMHDFilename + "CylinderPointCloud.txt");
			for (int i = 0; i < cylPoints->GetNumberOfPoints(); i++) {
				double* point = cylPoints->GetPoint(i);
				myfile << point[0] << " " << point[1] << " " << point[2] << "\n";
			}
			myfile.close();
			vtkNew<vtkPolyDataWriter> vtkWriter;
			std::string cylCropFilname = path + "Output/" + initialMHDFilename + "TestCylinderCrop.vtk";
			vtkWriter->SetFileName(cylCropFilname.c_str());
			vtkWriter->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
			vtkWriter->Write();
		}

		vtkNew<vtkSignedDistance> distance;
		vtkNew<vtkPCANormalEstimation> normals;
		vtkNew<vtkVertexGlyphFilter> glyphFilter;
		vtkNew<vtkSTLWriter> stlWriter;
		int sampleSize = 15;
		normals->SetInputData(polyData);
		normals->SetSampleSize(sampleSize);
		normals->SetNormalOrientationToGraphTraversal();
		normals->FlipNormalsOn();
		distance->SetInputConnection(normals->GetOutputPort());

		switch (reco) {
			case ExtractSurface: {
				double bounds[6];
				polyData->GetBounds(bounds);
				double range[3];
				for (int i = 0; i < 3; ++i)
				{
					range[i] = bounds[2 * i + 1] - bounds[2 * i];
				}
				glyphFilter->SetInputData(polyData);
				glyphFilter->Update();

				int dimension = 512;
				double radius;
				radius = std::max(std::max(range[0], range[1]), range[2]) / static_cast<double>(dimension) * 4; // ~4 voxels

				distance->SetRadius(radius);
				distance->SetDimensions(dimension, dimension, dimension);
				distance->SetBounds(bounds[0] - range[0] * .1, bounds[1] + range[0] * .1,
					bounds[2] - range[1] * .1, bounds[3] + range[1] * .1,
					bounds[4] - range[2] * .1, bounds[5] + range[2] * .1);
				vtkNew<vtkExtractSurface> surfaceExtract;
				surfaceExtract->SetInputConnection(distance->GetOutputPort());
				surfaceExtract->SetRadius(radius * .99);
				surfaceExtract->Update();
				targetPolyData->DeepCopy(surfaceExtract->GetOutput());
				stlFilename = path + "Output/ExtractSurface/" + initialMHDFilename + ".stl";
				compareFileName = path + "Output/ExtractSurface/" + initialMHDFilename + "Comp.vtk";
				errorFilename = path + "Output/ExtractSurface/" + initialMHDFilename + "Errors.txt";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surfaceExtract->GetOutputPort());
				stlWriter->Write();


				// Rotation tests (Robin)
				/*vtkNew<vtkTransform> transRef;
				transRef->Scale(1, 1, 1);
				transRef->Translate(0, 15, 0);
				transRef->RotateZ(-90);
				transRef->RotateX(90);
				vtkNew<vtkTransformPolyDataFilter> transformToRef;

				vtkNew<vtkPolyData> copyPolyDataNotRotated;
				copyPolyDataNotRotated->DeepCopy(targetPolyData);

				transformToRef->SetInputData(copyPolyDataNotRotated);
				transformToRef->SetTransform(transRef);
				transformToRef->Update();

				stlFilename = path + "Output/ExtractSurface/" + initialMHDFilename + "Rotated.stl";
				compareFileName = path + "Output/ExtractSurface/" + initialMHDFilename + "RotatedComp.vtk";
				errorFilename = path + "Output/ExtractSurface/" + initialMHDFilename + "RotatedErrors.txt";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(transformToRef->GetOutputPort());
				stlWriter->Write();*/

				// End of rotation test

				break;
			}
			case Poisson: {
				vtkSmartPointer<vtkPoissonReconstruction> surfacePois = vtkSmartPointer<vtkPoissonReconstruction>::New();
				surfacePois->SetDepth(14);
				surfacePois->SetInputConnection(normals->GetOutputPort());
				surfacePois->Update();
				targetPolyData->DeepCopy(surfacePois->GetOutput());
				stlFilename = path + "Output/Poisson/" + initialMHDFilename + ".stl";
				compareFileName = path + "Output/Poisson/" + initialMHDFilename + "Comp.vtk";
				errorFilename = path + "Output/Poisson/" + initialMHDFilename + "Errors.txt";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surfacePois->GetOutputPort());
				stlWriter->Write();
				break;
			}
			case PowerCrust: { // Marche pas sans subvox refinement
				try{
					vtkSmartPointer<vtkPowerCrustSurfaceReconstruction> surfacePowerCrust = vtkSmartPointer<vtkPowerCrustSurfaceReconstruction>::New();
					surfacePowerCrust->SetInputData(polyData);
					stlFilename = path + "Output/PowerCrust/" + initialMHDFilename + "PowerCrust.stl";
					compareFileName = path + "Output/PowerCrust/" + initialMHDFilename + "CompPowerCrust.vtk";
					errorFilename = path + "Output/PowerCrust/" + initialMHDFilename + "Errors.txt";
					surfacePowerCrust->Update();
					targetPolyData->DeepCopy(surfacePowerCrust->GetOutput());
					stlWriter->SetFileName(stlFilename.c_str());
					stlWriter->SetInputConnection(surfacePowerCrust->GetOutputPort());
					stlWriter->Write();
				}
				catch (itk::ExceptionObject& ex) {
					std::cout << ex.what() << std::endl;
				}
				break;
			}
			case SurfReconst: { // Super lent
				
				// Tests Robin : Reduce #vertices before surfacing

				/*vtkSmartPointer<vtkDecimatePro> decimateFilter = vtkSmartPointer<vtkDecimatePro>::New();
				decimateFilter->SetInputData(polyData);
				decimateFilter->SetTargetReduction(0.9);
				decimateFilter->PreserveTopologyOn();
				decimateFilter->Update();

				vtkNew<vtkPolyData> decimatedPolyData;
				decimatedPolyData->ShallowCopy(decimateFilter->GetOutput());

				vtkNew<vtkCleanPolyData> testCleaner;
				testCleaner->SetInputData(decimatedPolyData);
				stlWriter->SetFileName(path + "Output/SurfReconst/decimatedTest.stl");
				stlWriter->SetInputConnection(testCleaner->GetOutputPort());
				stlWriter->Write();*/
				
				/*
				std::cout << "Original points : " <<polyData->GetNumberOfPoints() << std::endl;
				vtkNew<vtkVertexGlyphFilter> vertexGlyphFilter;
				vertexGlyphFilter->SetInputData(polyData);
				vertexGlyphFilter->Update();

				vtkNew<vtkCleanPolyData> testCleaner;
				testCleaner->SetInputData(vertexGlyphFilter->GetOutput());
				testCleaner->SetTolerance(0.005);
				testCleaner->Update();
				std::cout << "Reduced points : " << testCleaner->GetOutput()->GetNumberOfPoints() << std::endl;

				vtkNew<vtkPLYWriter> plyWriter;
				plyWriter->SetFileName((path + "Output/SurfReconst/pointCloud.ply").c_str());
				plyWriter->SetInputData(polyData);
				plyWriter->Write();



				// Load the reduced point cloud obtained from the poisson disk filter from meshlab
				vtkNew<vtkPLYReader> plyReader;
				plyReader->SetFileName((path + "Output/SurfReconst/refPointCloud.ply").c_str());
				plyReader->Update();
				std::cout << plyReader->GetOutput()->GetNumberOfPoints() << std::endl;
				
				// End tests Robin*/

				std::cout << "Surface reconstruction : Start" << std::endl;


				vtkNew<vtkSurfaceReconstructionFilter> surf;
				surf->SetInputData(polyData);
				//surf->SetInputData(testCleaner->GetOutput());
				//surf->SetInputData(plyReader->GetOutput());
				surf->SetNeighborhoodSize(20);
				surf->SetSampleSpacing(0.1); // MODIFIED FROM 0.1
				std::cout << surf->GetSampleSpacing() << std::endl;

				vtkNew<vtkContourFilter> contourFilter;
				contourFilter->SetInputConnection(surf->GetOutputPort());
				contourFilter->SetValue(0, 0.0);

				// Sometimes the contouring algorithm can create a volume whose gradient
				// vector and ordering of polygon (using the right hand rule) are
				// inconsistent. vtkReverseSense cures this problem.
				vtkNew<vtkReverseSense> surface;
				surface->SetInputConnection(contourFilter->GetOutputPort());
				surface->ReverseCellsOn();
				surface->ReverseNormalsOn();
				surface->Update();
				targetPolyData->DeepCopy(surface->GetOutput());
				stlFilename = path + "Output/SurfReconst/" + initialMHDFilename + "SurfaceExtractFilter.stl";
				compareFileName = path + "Output/SurfReconst/" + initialMHDFilename + "CompSurfaceExtractFilter.vtk";
				errorFilename = path + "Output/SurfReconst/" + initialMHDFilename + "Errors.txt";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surface->GetOutputPort());
				stlWriter->Write();



				std::cout << "Surface reconstruction : End" << std::endl;

				break;
			}
		}
	}

	//Load reference STL
	vtkNew<vtkSTLReader> referenceReader;
	std::string referenceSTL = path + "Input/InitialModel.stl";
	//std::string referenceSTL = "C:/Users/DEV/Desktop/RobinStuff/Surface/build/Release/Input/InitialModel.stl";
	referenceReader->SetFileName(referenceSTL.c_str());
	referenceReader->Update();

	// Rotate reference (Robin) :

	// Circular : never transform anything

	// Poisson Helix : ((15, 0, 0)(90, 180, -90))

	// PowerCrust Helix : ((15, 0, 0)(90, 180, -90))

	// SurfaceReconstruction Helix : ((15, 0, 0)(90, 180, -90))

	// Surface Net Helix : ((-15, 0, 0)(90, 0, -90)) et changer dans l'algo mais je sais plus quoi

	// Flying Edges Helix : ((15, 0, 0)(90, 180, -90)) et changer 90 en -90 tout en laissant le scale

	

	vtkNew<vtkPolyData> test;
	test->DeepCopy(referenceReader->GetOutput());
	vtkNew<vtkTransform> transRef;
	transRef->Scale(1, 1, 1);
	if (geometry == 1) {
		if (reco == 0 || reco == 1 || reco == 2 || reco == 3 || reco == 4 || reco == 5) {
			transRef->Translate(15, 0, 0);
			transRef->RotateX(90); // -90 is the same by symmetry
			transRef->RotateY(180);
			transRef->RotateZ(-90);
		}
		/*if (reco == 4) {
			transRef->Translate(-15, 0, 0);
			transRef->RotateX(90); // -90 is the same by symmetry
			transRef->RotateY(0);
			transRef->RotateZ(-90);
		}*/
	}

	vtkNew<vtkTransform> transformBack; // Useful only for the computation on the edges at the end
	transformBack->RotateZ(90);
	transformBack->RotateY(180);
	transformBack->RotateX(-90);
	transformBack->Translate(-15, 0, 0);


	vtkNew<vtkTransformPolyDataFilter> transformToRef;
	vtkNew<vtkTransformPolyDataFilter> transformFromRef;


	transformToRef->SetInputData(test);
	transformToRef->SetTransform(transRef);
	transformToRef->Update();


	// Reverse the obtained polydata into volume (on the synthetic intial volume)
	// Make a copy just to be sure
	PolyDataToImageData(targetPolyData);

	// End rotate reference

	vtkNew<vtkCleanPolyData> referencePolyData;
	//referencePolyData->SetInputData(referenceReader->GetOutput());
	referencePolyData->SetInputData(transformToRef->GetOutput());
	std::cout << "Reference Loaded" << std::endl;


	vtkNew<vtkCleanPolyData> targetCleanPolyData;
	targetCleanPolyData->SetInputData(targetPolyData);

	// Align extracted mesh with reference mesh before comparing them

	vtkNew<vtkPolyData> alignTestPolyData;
	alignTestPolyData->DeepCopy(targetPolyData);

	vtkNew<vtkIterativeClosestPointTransform> icp;
	icp->SetSource(alignTestPolyData);
	icp->SetTarget(transformToRef->GetOutput());
	//icp->SetTarget(alignTestPolyData);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	//icp->GetLandmarkTransform()->SetModeToSimilarity();
	//icp->GetLandmarkTransform()->SetModeToAffine();
	icp->SetMaximumNumberOfIterations(500);
	icp->StartByMatchingCentroidsOff();
	icp->SetMaximumMeanDistance(.00001);
	icp->CheckMeanDistanceOn();
	icp->DebugOn();
	icp->Modified();
	icp->Update();

	vtkSmartPointer<vtkMatrix4x4> m = icp->GetMatrix();
	std::cout << "Resulting alignment matrix : " << *m << std::endl;

	vtkNew<vtkTransformPolyDataFilter> icpTransformFilter;
	icpTransformFilter->SetInputData(alignTestPolyData);
	icpTransformFilter->SetTransform(icp);
	icpTransformFilter->Update();
	targetCleanPolyData->SetInputData(icpTransformFilter->GetOutput());

	// Compute distance between vertices of the reconstructed surface and the reference mesh

	vtkNew<vtkDistancePolyDataFilter> distanceFilter;
	distanceFilter->SetInputConnection(1, referencePolyData->GetOutputPort());
	distanceFilter->SetInputConnection(0, targetCleanPolyData->GetOutputPort());
	distanceFilter->SetSignedDistance(false);
	distanceFilter->Update();

	vtkNew<vtkPolyDataWriter> vtkWriter;
	
	vtkWriter->SetFileName(compareFileName.c_str());
	vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
	vtkWriter->Write();

	// Write results as csv file

	vtkNew<vtkDelimitedTextWriter> vtkCSVWriter;
	vtkNew<vtkTable> table;
	vtkNew<vtkFloatArray> col;
	col->SetName("Distance");
	table->AddColumn(col);
	size_t nbPoints = distanceFilter->GetOutput()->GetNumberOfPoints();
	table->SetNumberOfRows(static_cast<vtkIdType>(nbPoints));
	double distanceValue[3];
	for (vtkIdType r = 0; r < table->GetNumberOfRows(); r++) {
		distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetTuple(r, distanceValue);
		table->SetValue(r, 0, distanceValue[0]);
	}
	std::string distanceFilename = "C:/Users/DEV/Desktop/RobinStuff/Spreadsheets/" + geometryString + algoToString[reco] + ".csv";
	vtkCSVWriter->SetFileName(distanceFilename.c_str());
	vtkCSVWriter->SetInputData(table);
	vtkCSVWriter->Update();



	// Tests stats Robin

	//computeDistanceStats(distanceFilter->GetOutputPort());
	/*vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputConnection(distanceFilter->GetOutputPort());
	mapper->SetScalarRange(
		distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetRange()[0],
		distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetRange()[1]);*/

	// Tests stats Robin


	// /!\ si computepointerror==true : le vtk de comparaison point à point est modifiée et meme la rotation ne marche plus


	// Mettre en fonction
	if (computePointError) {


		transformFromRef->SetInputData(targetPolyData);
		transformFromRef->SetTransform(transformBack); // Comment this and the line below out if in circular geometry
		transformFromRef->Update();					   //




		std::cout << "Compute point error" << std::endl;
		// Length of edges
		std::ofstream errorFile;
		errorFile.open(errorFilename);
		vtkNew<vtkStaticPointLocator> pointLocator;
		vtkPolyData* polyData = targetPolyData;
		//pointLocator->SetDataSet(polyData); // Roland version
		pointLocator->SetDataSet(transformFromRef->GetOutput());
		pointLocator->BuildLocator();
		double mean = 0;
		double min = 5000;
		double max = 0;
		double stddev = 0;

		// Hardcoded array because flemme
		double point1Coord[16][3] = { {-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},
									  {20.0, 20.0, 20.0},{20.0, 20.0, 20.0},{20.0, 20.0, -20.0},
									  {20.0, 20.0, -20.0},{-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},
									  {-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},{20.0, -20.0, 20.0},
									  {0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0} };

		double point2Coord[16][3] = { {20.0, 20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, -20.0},
									  {20.0, 20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
									  {-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
									  {20.0, -20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
									  {-20.0, 20.0, 20.0},{-20.0, 20.0, -20.0},{20.0, 20.0, -20.0},{20.0, 20.0, 20.0} };
		double distancesError[16];
		// Loop through all 16 outer edges
		for (size_t i = 0; i < 16; i++) {
			vtkIdType point1ID = pointLocator->FindClosestPoint(point1Coord[i]);
			double* point1Point = polyData->GetPoint(point1ID);
			double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
			vtkIdType point2ID = pointLocator->FindClosestPoint(point2Coord[i]);
			double* point2Point = polyData->GetPoint(point2ID);
			double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
			double distance = std::sqrt(vtkMath::Distance2BetweenPoints(point1, point2)); // don't forget to sqrt
			//compute distance btw the two points
			if (i < 12) {
				distancesError[i] = abs(distance - 40.0);
			}
			else {
				distancesError[i] = abs(distance - 41.2309); //length of the diagonal from the top
			}
			std::cout << distancesError[i] << " ";
			errorFile << distancesError[i] << " ";
			mean += distancesError[i];
			if (distancesError[i] < min)
				min = distancesError[i];
			if (distancesError[i] > max)
				max = distancesError[i];
		}
		mean = mean / 16.0;
		std::cout << std::endl;
		errorFile << "\n";
		for (size_t i = 0; i < 16; ++i) {
			stddev += pow(distancesError[i] - mean, 2);
		}
		errorFile << "Mean : " << mean << "\n";
		errorFile << "Min : " << min << "\n";
		errorFile << "Max : " << max << "\n";
		errorFile << "StdDev : " << stddev << "\n";
		// Angles
		double anglesErrors[3]; // Voir si j'en fais plus ?
		double realAngles[3] = { 90.0, 90.0, 43.313 };
		double point1Vec[4][3] = { {-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},
									  {-20.0, 20.0, 20.0} };
		double point2Vec[4][3] = { {20.0, -20.0, 20.0},{-20.0, -20.0, -20.0},{-20.0, 20.0, 20.0},
									  {0.0, 50.0, 0.0} };
		// Flemme de la boucle polalalalala
		std::vector<vtkVector3d> vectors;
			for (size_t i = 0; i < 4; i++) {
				vtkIdType point1ID = pointLocator->FindClosestPoint(point1Vec[i]);
				double* point1Point = polyData->GetPoint(point1ID);
				double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
				vtkIdType point2ID = pointLocator->FindClosestPoint(point2Vec[i]);
				double* point2Point = polyData->GetPoint(point2ID);
				double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
				double diff[3] = { point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2] };
				//create vector
				vtkVector3d vector = vtkVector3d(diff);
				vectors.push_back(vector);
			}

			anglesErrors[0] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[2].GetData())) - realAngles[0];
			anglesErrors[1] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[1].GetData())) - realAngles[1];
			anglesErrors[2] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[3].GetData(), vectors[2].GetData())) - realAngles[2];

		std::cout << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << std::endl;
		errorFile << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << "\n";


		// Edge error measurements Robin
		transformFromRef->SetInputData(distanceFilter->GetOutput());
		transformFromRef->SetTransform(transformBack); // Comment this and the line below out if in circular geometry
		transformFromRef->Update();					   //

		/*vtkWriter->SetFileName(path + "Output/ExtractSurface/rotationTest.vtk");
		vtkWriter->SetInputConnection(transformFromRef->GetOutputPort());
		vtkWriter->Write();*/
		

		std::ofstream edgeErrorFile;
		std::string edgeErrorFilename = errorFilename.substr(0, errorFilename.find(".txt"));
		edgeErrorFilename += "Edges.csv";
		edgeErrorFile.open(edgeErrorFilename);
		vtkNew<vtkStaticPointLocator> edgePointLocator;
		//edgePointLocator->SetDataSet(distanceFilter->GetOutput());
		edgePointLocator->SetDataSet(transformFromRef->GetOutput());
		edgePointLocator->BuildLocator();
		double edgeMean = 0;
		double edgeMin = 5000;
		double edgeMax = 0;
		double edgeStddev = 0;

		size_t nbPoints = 1000;

		double edgePointValue[3];

		for (size_t i = 0; i < 16; i++) {
			double stepX = (point2Coord[i][0] - point1Coord[i][0]) / nbPoints;
			double stepY = (point2Coord[i][1] - point1Coord[i][1]) / nbPoints;
			double stepZ = (point2Coord[i][2] - point1Coord[i][2]) / nbPoints;
			for (size_t p = 0; p < nbPoints; p++) {
				double edgePointIndices[3] = { point1Coord[i][0] + p * stepX, point1Coord[i][1] + p * stepY, point1Coord[i][2] + p * stepZ };
				//std::cout << "Edge point indices: " << edgePointIndices[0] << std::endl;
				vtkIdType edgePointID = edgePointLocator->FindClosestPoint(edgePointIndices);
				distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetTuple(edgePointID, edgePointValue);
				edgeMean += edgePointValue[0];

				if (edgePointValue[0] < edgeMin)
					edgeMin = edgePointValue[0];
				if (edgePointValue[0] > edgeMax)
					edgeMax = edgePointValue[0];
			}
		}
		edgeMean /= (nbPoints * 16);
		std::cout << "Edge points mean error : " << edgeMean << std::endl;

		for (size_t i = 0; i < 16; i++) {
			double stepX = (point2Coord[i][0] - point1Coord[i][0]) / nbPoints;
			double stepY = (point2Coord[i][1] - point1Coord[i][1]) / nbPoints;
			double stepZ = (point2Coord[i][2] - point1Coord[i][2]) / nbPoints;
			for (size_t p = 0; p < nbPoints; p++) {
				double edgePointIndices[3] = { point1Coord[i][0] + p * stepX, point1Coord[i][1] + p * stepY, point1Coord[i][2] + p * stepZ };
				vtkIdType edgePointID = edgePointLocator->FindClosestPoint(edgePointIndices);
				distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetTuple(edgePointID, edgePointValue);
				edgeStddev += pow(edgePointValue[0] - edgeMean, 2);
			}
		}
		edgeStddev /= (nbPoints * 16);
		edgeStddev = sqrt(edgeStddev);
		std::cout << "Edge points stddev : " << edgeStddev << std::endl;


		edgeErrorFile << "Mean,Min,Max,Std\n";
		edgeErrorFile << edgeMean << "," << edgeMin << "," << edgeMax << "," << edgeStddev << "\n";
		edgeErrorFile.close();



		// Cylinder
		vtkNew<vtkCylinder> cylinder;
		cylinder->SetCenter(-20.0, 0.0, 0.0);
		cylinder->SetRadius(10.15);
		cylinder->SetAxis(1.0, 0.0, 0.0);
		vtkNew<vtkImplicitBoolean> boolean;
		boolean->AddFunction(cylinder);
		vtkNew<vtkExtractPolyDataGeometry> extractPolyDataGeometry;
		extractPolyDataGeometry->SetInputData(polyData);
		extractPolyDataGeometry->SetExtractInside(true);
		extractPolyDataGeometry->SetImplicitFunction(boolean);
		extractPolyDataGeometry->Update();

		vtkNew<vtkConvertToPointCloud> pcConvert;
		pcConvert->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
		pcConvert->SetCellGenerationMode(vtkConvertToPointCloud::NO_CELLS);
		pcConvert->Update();
		vtkPolyData* cylPoints = pcConvert->GetOutput();
		std::ofstream myfile;
		myfile.open("PointCloud.txt");
		for (int i = 0; i < cylPoints->GetNumberOfPoints(); i++) {
			double* point = cylPoints->GetPoint(i);
			myfile << point[0] << " " << point[1] << " " << point[2] << "\n";
		}
		myfile.close();
		vtkNew<vtkPolyDataWriter> vtkWriter;
		vtkWriter->SetFileName((path + "Output/TestCylinderCrop.vtk").c_str());
		vtkWriter->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
		vtkWriter->Write();

		// Do the fitting -> https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf (banger)
		// Flemme de le coder je vais compiler GeometricTools à la place UwU

		// Section 7 shows how a cylinder can be fitted on 3D points

	}

	return EXIT_SUCCESS;
}
