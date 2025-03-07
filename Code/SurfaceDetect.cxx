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
//#include <vtkPoissonReconstruction.h>
#include <vtkCleanPolyData.h>
//#include <vtkPowerCrustSurfaceReconstruction.h>
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
#include "C:/Users/xris/Desktop/RobinStuff/Surface/build/spline.h"
#include <limits>
#include <vtkDelaunay2D.h>
#include <vtkDelaunay3D.h>
#include <vtkPolyVertex.h> 
#include <vtkCellData.h>
#include <vtkCellLocator.h>
#include <vtkCellCenters.h>
#include <vtkDoubleArray.h>
#include <vtkIdTypeArray.h>
#include <vtkCubeSource.h>
#include <vtkTriangleFilter.h>
#include <vtkMassProperties.h>
#include <vtkIntersectionPolyDataFilter.h>
#include <vtkSelectPolyData.h>
#include <vtkSelectEnclosedPoints.h>
#include "itkMultiplyImageFilter.h"


#include "C:/Users/xris/Desktop/RobinStuff/Surface/build/BilateralCannyEdgeDetectionImageFilter.h"


namespace constants {
	const double VOXEL_SIZE = 0.2;
	const double GAUSS_VARIANCE = 0.1;
	const double RANGE_VARIANCE = 5000;
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
	plyWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Output/PointCloudInitialVolume.ply");
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
	unsigned short inval = 65535;
	unsigned short outval = 0;
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
	writer->SetFileName("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/InitialVolumeFromPolyData.mhd");
	writer->SetInputData(imgstenc->GetOutput());
	writer->Write();
}

// Only used once to create our test volume from the voxelisation of a surface
void VoxelizeSurface() {
	//Load reference STL
	vtkNew<vtkSTLReader> referenceReader;
	//std::string referenceSTL = "C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Input/InitialModel.stl";
	//std::string referenceSTL = "C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Input/Cylinder.stl";
	//std::string referenceSTL = "C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Input/Cone.stl";
	std::string referenceSTL = "C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Input/Sphere.stl";
	referenceReader->SetFileName(referenceSTL.c_str());
	referenceReader->Update();

	vtkPolyData* polyData = referenceReader->GetOutput();

	double bounds[6];
	polyData->GetBounds(bounds);
	double spacing[3];
	spacing[0] = constants::VOXEL_SIZE;
	spacing[1] = constants::VOXEL_SIZE;
	spacing[2] = constants::VOXEL_SIZE;

	double padding = 10;

	vtkNew<vtkImageData> voxelizedImage;
	voxelizedImage->SetSpacing(spacing);
	int dim[3];
	for (int i = 0; i < 3; i++) {
		dim[i] = static_cast<int>(ceil(bounds[i * 2 + 1] - bounds[i * 2] + padding) / spacing[i]);
	}
	voxelizedImage->SetDimensions(dim);
	voxelizedImage->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);


	double origin[3];
	origin[0] = bounds[0] - padding / 2 + spacing[0] / 2;
	origin[1] = bounds[2] - padding / 2 + spacing[1] / 2;
	origin[2] = bounds[4] - padding / 2 + spacing[2] / 2;
	voxelizedImage->SetOrigin(origin);
	voxelizedImage->AllocateScalars(VTK_UNSIGNED_SHORT, 1);
	unsigned short inval = 65535;
	unsigned short outval = 0;
	unsigned short midval = 20000;
	vtkIdType count = voxelizedImage->GetNumberOfPoints();
	vtkIdType cellCount = voxelizedImage->GetNumberOfCells();
	for (vtkIdType i = 0; i < count; i++) {
		voxelizedImage->GetPointData()->GetScalars()->SetTuple1(i, midval);
	}


	vtkPolyData* voxels = vtkPolyData::New();
	vtkPoints* points = vtkPoints::New();
	auto vertexes = vtkSmartPointer<vtkCellArray>::New();


	for (double z = origin[2]; z < bounds[5] + padding / 2 - spacing[2] / 2; z += spacing[2]) {
		for (double y = origin[1]; y < bounds[3] + padding / 2 - spacing[1] / 2; y += spacing[1]) {
			for (double x = origin[0]; x < bounds[1] + padding / 2 - spacing[0] / 2; x += spacing[0]) {
				double coord[3];
				coord[0] = x; coord[1] = y; coord[2] = z;
				points->InsertNextPoint(coord);
			}
		}
	}

	double maxVoxelRadius = constants::VOXEL_SIZE * std::sqrt(3) / 2;


	vtkNew<vtkCellLocator> cellLocator;
	cellLocator->SetDataSet(polyData);
	cellLocator->BuildLocator();
	double closestPointOnCell[3];
	double closestPointDist2;
	vtkIdType closestCellId;
	int subId; // useless ?

	vtkNew<vtkImplicitPolyDataDistance> implicitPolyDataDistance;
	implicitPolyDataDistance->SetInput(polyData);

	vtkNew<vtkFloatArray> signedDistances; // using signed distances directly is clever because negative distance means it's inside, no need to work with normals
	signedDistances->SetNumberOfComponents(1);
	signedDistances->SetName("SignedDistances");

	// Evaluate the signed distance function at all of the grid points
	for (vtkIdType voxelId = 0; voxelId < points->GetNumberOfPoints(); ++voxelId)
	{
		double p[3];
		double closestPoint[3];
		points->GetPoint(voxelId, p);
		float signedDistance = implicitPolyDataDistance->EvaluateFunctionAndGetClosestPoint(p, closestPoint);
		signedDistances->InsertNextValue(signedDistance);


		//cellLocator->FindClosestPoint(closestPoint, closestPointOnCell, closestCellId, subId, closestPointDist2);
		if (signedDistances->GetValue(voxelId) > maxVoxelRadius) {
			voxelizedImage->GetPointData()->GetScalars()->SetTuple1(voxelId, outval);
		}
		else if (signedDistances->GetValue(voxelId) < -maxVoxelRadius) {
			voxelizedImage->GetPointData()->GetScalars()->SetTuple1(voxelId, inval);
		}
	}

	vtkNew<vtkPolyData> distancePolyData;
	distancePolyData->SetPoints(points);
	distancePolyData->GetPointData()->SetScalars(signedDistances);





	// Test Cube volume
	vtkNew<vtkCubeSource> smallCubeSource;
	double centerCoordSmallCube[3] = { 5.0 , 5.0, 7.07 };
	smallCubeSource->SetCenter(centerCoordSmallCube);
	smallCubeSource->SetXLength(0.2);
	smallCubeSource->SetYLength(0.2);
	smallCubeSource->SetZLength(0.2);
	smallCubeSource->Update();

	vtkNew<vtkCubeSource> bigCubeSource;
	double centerCoordBigCube[3] = { 0 , 1, 0 };
	bigCubeSource->SetCenter(centerCoordBigCube);
	bigCubeSource->SetXLength(4);
	bigCubeSource->SetYLength(4);
	bigCubeSource->SetZLength(4);
	bigCubeSource->Update();



	vtkNew<vtkTriangleFilter> triangleFilterForSmallCube;
	triangleFilterForSmallCube->SetInputConnection(smallCubeSource->GetOutputPort());
	vtkNew<vtkTriangleFilter> triangleFilterForBigCube;
	triangleFilterForBigCube->SetInputConnection(bigCubeSource->GetOutputPort());

	vtkNew<vtkIntersectionPolyDataFilter> intersectionPolyDataFilter;
	intersectionPolyDataFilter->SetInputConnection(0,triangleFilterForSmallCube->GetOutputPort());
	intersectionPolyDataFilter->SetInputConnection(1, referenceReader->GetOutputPort());
	intersectionPolyDataFilter->SplitFirstOutputOn();
	intersectionPolyDataFilter->SplitSecondOutputOn();
	intersectionPolyDataFilter->CheckMeshOn();
	intersectionPolyDataFilter->ComputeIntersectionPointArrayOn();
	intersectionPolyDataFilter->CheckInputOn();
	intersectionPolyDataFilter->Update();
	std::cout << "Number of intersections between the two polydatas : " << intersectionPolyDataFilter->GetNumberOfIntersectionPoints() << std::endl;
	std::cout << "Intersection done." << std::endl;

	//intersectionPolyDataFilter->PrintSelf(std::cout, vtkIndent(0));


	/*vtkNew<vtkSelectPolyData> selectionFilter;
	selectionFilter->SetInputConnection(triangleFilterForBigCube->GetOutputPort());
	selectionFilter->SetLoop(intersectionPolyDataFilter->GetOutput()->GetPoints());
	selectionFilter->GenerateSelectionScalarsOn();
	selectionFilter->SetSelectionModeToLargestRegion();
	selectionFilter->SetEdgeSearchModeToDijkstra();

	vtkNew<vtkClipPolyData> clipFilter;
	clipFilter->SetInputConnection(selectionFilter->GetOutputPort());
	clipFilter->Update();*/




	// The "remeshed" output from the vtkIntersectionPolyDataFilter does not yield the expected result. Let's try to work around it
	vtkNew<vtkPolyData> modCube;
	vtkNew<vtkPoints> modCubePoints;
	vtkNew<vtkCellArray> modCubeCells;
	
	/*modCube->DeepCopy(intersectionPolyDataFilter->GetOutput(1));
	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(2)->GetNumberOfPoints(); i++) {
		double p[3];
		intersectionPolyDataFilter->GetOutput(2)->GetPoint(i, p);
		modCube->GetPoints()->InsertNextPoint(p);
	}*/

	std::map<std::vector<double>, vtkIdType[3]> coordToIndex;
	vtkIdType newPointIndex = 0;

	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(1)->GetNumberOfPoints(); i++) {
		if (intersectionPolyDataFilter->GetOutput(1)->GetPointData()->GetArray("BoundaryPoints")->GetTuple(i)[0]) {
			double *coords = intersectionPolyDataFilter->GetOutput(1)->GetPoints()->GetPoint(i);
			std::vector<double> coordsVector; coordsVector.push_back(coords[0]); coordsVector.push_back(coords[1]); coordsVector.push_back(coords[2]);
			coordToIndex[coordsVector][0] = i;
			coordToIndex[coordsVector][2] = newPointIndex;
			newPointIndex++;
			modCubePoints->InsertNextPoint(intersectionPolyDataFilter->GetOutput(1)->GetPoint(i));
		}
	}
	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(2)->GetNumberOfPoints(); i++) { // Useless as it will extract the same points as above since they are the points common to both surfaces
		if (intersectionPolyDataFilter->GetOutput(2)->GetPointData()->GetArray("BoundaryPoints")->GetTuple(i)[0]) {
			double* coords = intersectionPolyDataFilter->GetOutput(2)->GetPoints()->GetPoint(i);
			std::vector<double> coordsVector; coordsVector.push_back(coords[0]); coordsVector.push_back(coords[1]); coordsVector.push_back(coords[2]);
			coordToIndex[coordsVector][1] = i;
			//modCubePoints->InsertNextPoint(intersectionPolyDataFilter->GetOutput(2)->GetPoint(i));
		}
	}

	// Points must be at the same index in both polys in order to merge the cells correctly

	// Test the hashmap : (it works)
	/*for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(1)->GetNumberOfPoints(); i++) {
		if (intersectionPolyDataFilter->GetOutput(1)->GetPointData()->GetArray("BoundaryPoints")->GetTuple(i)[0]) {
			double* coords = intersectionPolyDataFilter->GetOutput(1)->GetPoints()->GetPoint(i);
			std::vector<double> coordsVector; coordsVector.push_back(coords[0]); coordsVector.push_back(coords[1]); coordsVector.push_back(coords[2]);
			std::cout << "Point " << i << " | Index1 : " << coordToIndex[coordsVector][0] << " Index 2 : " << coordToIndex[coordsVector][1] << std::endl;
		}
	}*/


	// Filter which points are enclosed 
	vtkNew<vtkSelectEnclosedPoints> enclosedPointsSmallInBig;
	enclosedPointsSmallInBig->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(1));
	enclosedPointsSmallInBig->SetSurfaceConnection(intersectionPolyDataFilter->GetOutputPort(2));
	enclosedPointsSmallInBig->Update();

	vtkNew<vtkSelectEnclosedPoints> enclosedPointsBigInSmall;
	enclosedPointsBigInSmall->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(2));
	enclosedPointsBigInSmall->SetSurfaceConnection(intersectionPolyDataFilter->GetOutputPort(1));
	enclosedPointsBigInSmall->Update();


	// First copy the cells of the first poly.
	bool allPointsOnIntersection;
	bool allPointsInsideOtherPolyData;
	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(1)->GetNumberOfCells(); i++) {
		allPointsOnIntersection = true;
		allPointsInsideOtherPolyData = true;
		vtkNew<vtkIdList> pointIndexes;
		pointIndexes->SetNumberOfIds(3); // Valid as long as we are only working with triangles
		for (vtkIdType j = 0; j < intersectionPolyDataFilter->GetOutput(1)->GetCell(i)->GetNumberOfPoints(); j++) {
			double *coords = intersectionPolyDataFilter->GetOutput(1)->GetCell(i)->GetPoints()->GetPoint(j);
			std::vector<double> coordsVector; coordsVector.push_back(coords[0]); coordsVector.push_back(coords[1]); coordsVector.push_back(coords[2]);
			if (coordToIndex.find(coordsVector) == coordToIndex.end()) { // Meaning the point is not on the intersection
				allPointsOnIntersection = false;
				/*vtkNew<vtkPolyData> tmpPolyData;
				vtkNew<vtkPoints> tmpPoints;
				tmpPoints->InsertNextPoint(coords);
				tmpPolyData->SetPoints(tmpPoints);
				enclosedPointsSmallInBig->SetInputData(tmpPolyData); // EXTREEEEEEEMELY SLOW, must be changed because we should instead find a way to retrieve the point index j in the global enclosed filter
				enclosedPointsSmallInBig->Update();
				if (!enclosedPointsSmallInBig->IsInside(0)) {
					allPointsInsideOtherPolyData = false;
				}
				else {
					//pointIndexes->SetId(j, index) // Once again, we need the index.... And the point is not even set 
				}*/
			}
			else {
				pointIndexes->SetId(j, coordToIndex[coordsVector][2]);
			}
		}
		if (allPointsOnIntersection) { // All three points of the cell are on the intersection, we can add the cell.
			modCubeCells->InsertNextCell(pointIndexes);
			//modCubeCells->InsertNextCell(intersectionPolyDataFilter->GetOutput(1)->GetCell(i)); // It's wrong to use index i here because in fact, there are less points here than on the first polydata. We must instead recreate a brand new cell.
			//std::cout << "Added Cell " << i << std::endl;
		}
	}


	// Then copy the cells of the second poly.
	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(2)->GetNumberOfCells(); i++) {
		allPointsOnIntersection = true;
		vtkNew<vtkIdList> pointIndexes;
		pointIndexes->SetNumberOfIds(3); // Valid as long as we are only working with triangles
		for (vtkIdType j = 0; j < intersectionPolyDataFilter->GetOutput(2)->GetCell(i)->GetNumberOfPoints(); j++) {
			double* coords = intersectionPolyDataFilter->GetOutput(2)->GetCell(i)->GetPoints()->GetPoint(j);
			std::vector<double> coordsVector; coordsVector.push_back(coords[0]); coordsVector.push_back(coords[1]); coordsVector.push_back(coords[2]);
			if (coordToIndex.find(coordsVector) == coordToIndex.end()) { // Meaning the point is not on the intersection
				allPointsOnIntersection = false;
			}
			else {
				pointIndexes->SetId(j, coordToIndex[coordsVector][2]);
			}
		}
		if (allPointsOnIntersection) { // All three points of the cell are on the intersection, we can add the cell.
			modCubeCells->InsertNextCell(pointIndexes);
			//modCubeCells->InsertNextCell(intersectionPolyDataFilter->GetOutput(1)->GetCell(i)); // It's wrong to use index i here because in fact, there are less points here than on the first polydata. We must instead recreate a brand new cell.
			//std::cout << "Added Cell " << i << std::endl;
		}
	}




	


	/*auto insideArray = dynamic_cast<vtkDataArray*>(selectEnclosedPoints->GetOutput()->GetPointData()->GetArray("SelectedPoints"));
	for (vtkIdType i = 0; i < insideArray->GetNumberOfTuples(); i++) {
		std::cout << "Tuple " << i << "(" << selectEnclosedPoints->GetOutput()->GetPoint(i)[0] << ", " << selectEnclosedPoints->GetOutput()->GetPoint(i)[1] << ", " << selectEnclosedPoints->GetOutput()->GetPoint(i)[2] << "): ";
		if (insideArray->GetComponent(i, 0) == 1) { std::cout << "inside" << std::endl; }
		else { std::cout << "outside" << std::endl; }
		std::cout << "Real point : (" << intersectionPolyDataFilter->GetOutput(1)->GetPoint(i)[0] << ", " << intersectionPolyDataFilter->GetOutput(1)->GetPoint(i)[1] << ", " << intersectionPolyDataFilter->GetOutput(1)->GetPoint(i)[2] << ")" << std::endl;
	}
	std::cout << std::endl;

	vtkNew<vtkPolyData> pointTestPolyData;
	pointTestPolyData->DeepCopy(intersectionPolyDataFilter->GetOutput(1));
	vtkSmartPointer<vtkIntArray> isInside = vtkSmartPointer<vtkIntArray>::New();
	isInside->SetName("isInside");
	isInside->SetNumberOfComponents(1);

	for (vtkIdType i = 0; i < intersectionPolyDataFilter->GetOutput(1)->GetNumberOfPoints(); i++) {
		isInside->InsertNextTuple1(selectEnclosedPoints->IsInside(i));
	}
	pointTestPolyData->GetPointData()->SetScalars(isInside);*/
	



	modCube->SetPoints(modCubePoints);
	modCube->SetPolys(modCubeCells);


	vtkNew<vtkTriangleFilter> resultTriangleFilter;
	resultTriangleFilter->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(1));

	vtkNew<vtkPolyDataNormals> normals;
	normals->SetInputConnection(resultTriangleFilter->GetOutputPort());
	normals->ConsistencyOn();
	normals->SplittingOff();

	vtkNew<vtkMassProperties> massProperties;
	massProperties->SetInputConnection(normals->GetOutputPort());
	massProperties->Update();
	std::cout << "Volume: " << massProperties->GetVolume() << std::endl;

	vtkNew<vtkPolyDataWriter> vtkWriter;
	vtkWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Output/smallObject.vtk");
	vtkWriter->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(1));
	vtkWriter->Write();
	vtkWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Output/bigObject.vtk");
	vtkWriter->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(2));
	vtkWriter->Write();
	vtkWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Output/modCube.vtk");
	vtkWriter->SetInputData(modCube);
	vtkWriter->Write();

	vtkNew<vtkMetaImageWriter> imageWriter;
	imageWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Output/InitialVolumeFromPolyData.mhd");
	imageWriter->SetInputData(voxelizedImage);
	imageWriter->Write();

	vtkNew<vtkPLYWriter> plyWriter;
	plyWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Non-Binary_Voxelization/build/Release/Output/modCube.ply");
	plyWriter->SetInputData(modCube);
	plyWriter->Write();



	// Rendering tests

	vtkNew<vtkNamedColors> colors;
	vtkNew<vtkPolyDataMapper> mapper1;
	vtkNew<vtkPolyDataMapper> mapper2;
	vtkNew<vtkPolyDataMapper> intersectionMapper;
	intersectionMapper->SetInputConnection(intersectionPolyDataFilter->GetOutputPort());
	intersectionMapper->ScalarVisibilityOff();
	vtkNew<vtkActor> intersectionActor;
	intersectionActor->SetMapper(intersectionMapper);
	intersectionActor->GetProperty()->SetColor(colors->GetColor3d("White").GetData());
	mapper1->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(1));
	mapper1->ScalarVisibilityOff();
	vtkNew<vtkActor> actor1;
	actor1->SetMapper(mapper1);
	actor1->GetProperty()->SetOpacity(.3);
	actor1->GetProperty()->SetColor(colors->GetColor3d("Lime").GetData());
	mapper2->SetInputConnection(intersectionPolyDataFilter->GetOutputPort(2));
	vtkNew<vtkActor> actor2;
	actor2->SetMapper(mapper2);
	actor2->GetProperty()->SetOpacity(0.3);
	actor2->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());
	mapper1->ScalarVisibilityOff();
	vtkNew<vtkRenderer> renderer;
	renderer->AddViewProp(actor1);
	renderer->AddViewProp(actor2);
	renderer->AddViewProp(intersectionActor);
	renderer->SetBackground(colors->GetColor3d("SlateGray").GetData());
	vtkNew<vtkRenderWindow> renderWindow;
	renderWindow->AddRenderer(renderer);
	renderWindow->SetWindowName("Intersection with small cube");
	vtkNew<vtkRenderWindowInteractor> renWinInteractor;
	renWinInteractor->SetRenderWindow(renderWindow);
	renderWindow->Render();
	renWinInteractor->Start();



	points->Delete();
	voxels->Delete();


	//CubeTest();
}






double vectorNorm(itk::Index<3> vector) {
	long x = vector[0], y = vector[1], z = vector[2];
	double norm = x*x + y*y + z*z;
	norm = sqrt(norm);
	return norm;
}

double findMaxGradientOffset(tk::spline s, float norm, double x0) {
	double epsilon = constants::VOXEL_SIZE / 20;

	double maxGradient = 0;
	double maxX = x0;
	// On cherche uniquement entre le voxel max et ses 2 voisins, mais on pourrait extend à d'autres voxels
	for (double x = x0 - constants::VOXEL_SIZE * norm; x <= x0 + constants::VOXEL_SIZE * norm; x += epsilon) {
		//std::cout << "Spline deriv value : " << s.deriv(1, x) << " S'' value : " << s.deriv(2, x) << " X : " << x << " Spline value : " << s(x) << std::endl;
		if (abs(s(x)) > maxGradient) {
			maxGradient = abs(s(x));
			maxX = x;
		}
	}
	return maxX;
}

void printSpline(tk::spline s, float norm, double x0, double window_size) {
	double epsilon = constants::VOXEL_SIZE / 20;

	double maxGradient = 0;
	double maxX = x0;
	// On cherche uniquement entre le voxel max et ses 2 voisins, mais on pourrait extend à d'autres voxels
	for (double x = x0 - constants::VOXEL_SIZE * norm * window_size; x <= x0 + constants::VOXEL_SIZE * norm * window_size; x += epsilon) {
		std::cout << "(" << x << ") : " << s(x) << "   ";
		//std::cout << "Spline deriv value : " << s.deriv(1, x) << " S'' value : " << s.deriv(2, x) << " X : " << x << " Spline value : " << s(x) << std::endl;
		/*if (abs(s(x)) > maxGradient) {
			maxGradient = abs(s(x));
			maxX = x;
		}*/
	}
	std::cout << std::endl;
}

// Useless function. Delete eventually
/*void convertSphToCartesian(double* offsetX, double* offsetY, double* offsetZ, double norm, itk::Index<3> dir, double offsetRho) {
	double x = dir[0], y = dir[1], z = dir[2];

	double r = norm;
	double theta = acos(z / r);
	double phi = ((double(0) < y) - (y < double(0))) * acos(x / hypot(x, y));

	*offsetX = offsetRho * sin(theta) * cos(phi); if (std::isnan(*offsetX)) *offsetX = 0;
	*offsetY = offsetRho * sin(theta) * sin(phi); if (std::isnan(*offsetY)) *offsetY = 0;
	*offsetZ = offsetRho * cos(theta); if (std::isnan(*offsetZ)) *offsetZ = 0;
}*/

void offsetToCartesian(double* offsetX, double* offsetY, double* offsetZ, double offset, itk::Index<3> direction) {
	*offsetX = offset * direction[0];
	*offsetY = offset * direction[1];
	*offsetZ = offset * direction[2];
}

double centralDifference(itk::Image<unsigned short, 3>::Pointer image, itk::Index<3> voxelIndex, itk::Index<3> direction, itk::Image<unsigned short, 3>::RegionType region) {

	long i = direction[0], j = direction[1], k = direction[2];
	double norm = vectorNorm(direction);
	long x = voxelIndex[0], y = voxelIndex[1], z = voxelIndex[2];
	double previousGradientValue, nextGradientValue;

	itk::Index<3> nextIndex = { {x + i, y + j, z + k} };
	itk::Index<3> previousIndex = { {x - i, y - j, z - k} };
	

	region.IsInside(previousIndex) ? previousGradientValue = image->GetPixel(previousIndex) : previousGradientValue = 0;
	region.IsInside(nextIndex) ? nextGradientValue = image->GetPixel(nextIndex) : nextGradientValue = 0;

	float dx = (nextGradientValue - previousGradientValue)
		/ (2 * norm * constants::VOXEL_SIZE);


	/*if (x == 0 && y == 250 && z == 101) {
		std::cout << "Next neighbour gray value : " << image->GetPixel(nextIndex) << " Previous neighbour gray value : " << image->GetPixel(previousIndex) << std::endl;
		std::cout << "Gradient value in (0, 250, 101) : " << dx << std::endl << std::endl;
	}
	if (x == 1 && y == 250 && z == 101) {
		std::cout << "Next neighbour gray value : " << image->GetPixel(nextIndex) << " Previous neighbour gray value : " << image->GetPixel(previousIndex) << std::endl;
		std::cout << "Gradient value in (1, 250, 101) : " << dx << std::endl << std::endl;
	}
	if (x == 2 && y == 250 && z == 101) {
		std::cout << "Next neighbour gray value : " << image->GetPixel(nextIndex) << " Previous neighbour gray value : " << image->GetPixel(previousIndex) << std::endl;
		std::cout << "Gradient value in (2, 250, 101) : " << dx << std::endl << std::endl;
	}*/


	return dx;
}


int main(int argc, char** argv)
{
	// Remove the following line, it's only for testing.
	//VoxelizeSurface();
	//return 0;

	std::string path = "C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/";

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
	enum recoAlgoEnum { ExtractSurface, Poisson, PowerCrust, SurfReconst, SurfaceNets, FlyingEdges, Delaunay};
	recoAlgoEnum reco = ExtractSurface;
	std::map<recoAlgoEnum, std::string> algoToString = {
		{ExtractSurface, "ExtractSurface"}, {Poisson, "Poisson"}, {PowerCrust, "PowerCrust"}, {SurfReconst, "SurfReconst"}, {SurfaceNets, "SurfaceNets"}, {FlyingEdges, "FlyingEdges"}, {Delaunay, "Delaunay"}
	};

    std::string initialMHDFilename = "volRolandHelix";

    	std::cout << "volRolandHelix" << std::endl;
		std::cout.flush();

	if (initialMHDFilename == "Reference")
		cropPossibleGradientSearchSpace = false;


	// Robin parameters
	bool centerOfGravity = false;
	bool gradientBasedSubVox = true;

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
		image = itk::ReadImage<InputImageType>("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Input/" + initialMHDFilename + ".mhd");
		//image = itk::ReadImage<InputImageType>("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Input/volRolandHelix.mhd");
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


	// Load theoretical images
	if (initialMHDFilename == "simple_cube" || initialMHDFilename == "cylinder005" || initialMHDFilename == "cylinder01" || initialMHDFilename == "cylinder02" || initialMHDFilename == "rotated_cube02" || initialMHDFilename == "rotated_cube01" || initialMHDFilename == "rotated_cube005") {
		CannyOutputImageType::Pointer theoreticalImage = itk::ReadImage<CannyOutputImageType>("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Input/" + initialMHDFilename + ".mhd");
		
		using MultiplyFilterType = itk::MultiplyImageFilter<CannyOutputImageType, CannyOutputImageType, CannyOutputImageType>;
		auto multiplyFilter = MultiplyFilterType::New();
		multiplyFilter->SetInput(theoreticalImage);
		multiplyFilter->SetConstant(65535);
		multiplyFilter->Update();

		using CastFilterType = itk::CastImageFilter<CannyOutputImageType, InputImageType>;
		auto castFilter = CastFilterType::New();
		castFilter->SetInput(multiplyFilter->GetOutput());
		image = castFilter->GetOutput();

		typedef itk::ImageFileWriter<InputImageType> ImageWriter;
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(image);
		imageWriter->SetFileName("./Output/TestSphere.mhd");
		imageWriter->Update();
		// The bug lies in this function
		std::cout << "Working with simple geometric shapes" << std::endl;
	}


	// Fin tests Robin

	InputImageType::RegionType region = image->GetLargestPossibleRegion();
	InputImageType::SizeType size = region.GetSize();
	std::cout << "Size : " << size << std::endl;

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
	typedef itk::BilateralCannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> BilateralCannyFilter;
	BilateralCannyFilter::Pointer bilateralCanny = BilateralCannyFilter::New();
	bilateralCanny->SetInput(castFilter->GetOutput());
	bilateralCanny->SetDomainSigmas(constants::GAUSS_VARIANCE);
	bilateralCanny->SetRangeSigma(constants::RANGE_VARIANCE);
	if (initialMHDFilename == "Reference") {
		bilateralCanny->SetLowerThreshold(0.1f);
		bilateralCanny->SetUpperThreshold(0.9f);
	}
	else {
		bilateralCanny->SetLowerThreshold(800.f); // 800
		bilateralCanny->SetUpperThreshold(2500.f); // 2500
	}


	
	typedef itk::CannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> CannyFilter;
	CannyFilter::Pointer canny = CannyFilter::New();
	canny->SetInput(castFilter->GetOutput());
	if (initialMHDFilename == "Reference") {
		canny->SetLowerThreshold(0.3f);
		canny->SetUpperThreshold(0.8f);
	}
	else {
		canny->SetLowerThreshold(800.f); // 800
		canny->SetUpperThreshold(2500.f); // 2500
	}
	
	std::cout << "Threshold : " << canny->GetLowerThreshold() << " " << canny->GetUpperThreshold() << " " << canny->GetMaximumError() << std::endl;
	canny->SetVariance(constants::GAUSS_VARIANCE);

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
		imageWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/TestCanny.mhd");
		imageWriter->Update();
		std::cout << "Canny write done !" << std::endl;
	}
	std::string stlFilename;
	std::string compareFileName;
	std::string errorFilename;
	vtkNew<vtkPolyData> targetPolyData;
	vtkNew<vtkPolyData> polyDataBeforeMeshing;

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

		vtkNew<vtkPoints> points;

		if (!centerOfGravity && !gradientBasedSubVox) {
			typedef itk::Index<3> indexType;
			// On ne fait plus le centre de gravité on fait directement le rafinment sous voxelique sur les centre des points de Canny
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
			}
		}

		else if (centerOfGravity) { // In the end we we decided it's bad 
			std::cout << "STEP : begin center of gravity" << std::endl;

			// Generate point cloud based on the center of gravity technique instead of directly using the voxels positions given by Canny
			// One goal is to try different window sizes to find the optimal one.
			// 

			/*auto gradientFilterForSubVox = GradientFilterType::New();
			gradientFilterForSubVox->SetInput(gaussianFilter->GetOutput());
			gradientFilterForSubVox->Update();
			auto gradientVecImageForSubVox = gradientFilterForSubVox->GetOutput();

			unsigned short window_size = 1;
			typedef itk::Index<3> indexType;
			for (size_t i = 0; i < size[0]; i++) {
				for (size_t j = 0; j < size[1]; j++) {
					for (size_t k = 0; k < size[2]; k++) {
						indexType currIndex;
						indexType index;
						currIndex[0] = i;
						currIndex[1] = j;
						currIndex[2] = k;

						if (cannyImage->GetPixel(currIndex) != 0.0f) {

							float denominatorSum = 0;
							float numeratorSum = 0;
							for (int l = i - window_size; l <= i + window_size; l++) {
								if (i >= 0 && i < size[0]) {
									index[0] = l;
									index[1] = j;
									index[2] = k;
									auto currGradient = gradientVecImageForSubVox->GetPixel(index)[0];
									//if (abs(gradientVecImageForSubVox->GetPixel(currIndex)[0]) >= abs(currGradient))
									//{
										//numeratorSum += n * image->GetPixel(index);
										//denominatorSum += image->GetPixel(index);
										numeratorSum += l * abs(currGradient);
										denominatorSum += abs(currGradient);
									//}
									//std::cout << "Canny gradient : " << gradientVecImage->GetPixel(currIndex)[0] << " Neighbour gradient "<< index << " : " << currGradient << std::endl;
								}
							}
							float absPosX;
							if (denominatorSum == 0)
								absPosX = (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0];
							else {
								absPosX = (numeratorSum / denominatorSum) * image->GetSpacing()[0] + image->GetOrigin()[0];
								//std::cout << "Index : " << currIndex << std::endl;
								//std::cout << "OG pos X : " << (currIndex[0]) * image->GetSpacing()[0] + image->GetOrigin()[0] << std::endl;
								//std::cout << "SubVox pos X : " << absPosX << std::endl << std::endl;
							}

							denominatorSum = 0;
							numeratorSum = 0;
							for (int m = j - window_size; m <= j + window_size; m++) {
								if (j >= 0 && j < size[1]) {
									index[0] = i;
									index[1] = m;
									index[2] = k;
									auto currGradient = gradientVecImageForSubVox->GetPixel(index)[1];
									//if (abs(gradientVecImageForSubVox->GetPixel(currIndex)[1]) >= abs(currGradient))
									//{
										//numeratorSum += m * image->GetPixel(index);
										//denominatorSum += image->GetPixel(index);
										numeratorSum += m * abs(currGradient);
										denominatorSum += abs(currGradient);
									//}
									//std::cout << "Canny gradient : " << gradientVecImage->GetPixel(currIndex)[0] << " Neighbour gradient " << index << " : " << currGradient << std::endl;
								}
							}
							float absPosY;
							if (denominatorSum == 0)
								absPosY = -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]);
							else {
								absPosY = -((numeratorSum / denominatorSum) * image->GetSpacing()[1] + image->GetOrigin()[1]);
								//std::cout << "OG pos Y : " << -((currIndex[1]) * image->GetSpacing()[1] + image->GetOrigin()[1]) << std::endl;
								//std::cout << "SubVox pos Y : " << absPosY << std::endl << std::endl;
							}

							denominatorSum = 0;
							numeratorSum = 0;
							for (int n = k - window_size; n <= k + window_size; n++) {
								if (k >= 0 && k < size[2]) {
									index[0] = i;
									index[1] = j;
									index[2] = n;
									float currGradient = gradientVecImageForSubVox->GetPixel(index)[2];
									//if (abs(gradientVecImageForSubVox->GetPixel(currIndex)[2]) >= abs(currGradient))
									//{
										//numeratorSum += n * image->GetPixel(index);
										//denominatorSum += image->GetPixel(index);
										numeratorSum += n * abs(currGradient);
										denominatorSum += abs(currGradient);
									//}
									//std::cout << "Canny gradient : " << gradientVecImage->GetPixel(currIndex)[0] << " Neighbour gradient " << index << " : " << currGradient << std::endl;
								}
							}
							float absPosZ;
							if (denominatorSum == 0)
								absPosZ = (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2];
							else {
								absPosZ = (numeratorSum / denominatorSum) * image->GetSpacing()[2] + image->GetOrigin()[2];
								//std::cout << "OG pos Z : " << (currIndex[2]) * image->GetSpacing()[2] + image->GetOrigin()[2] << std::endl;
								//std::cout << "SubVox pos Z : " << absPosZ << std::endl << std::endl;
							}

							points->InsertNextPoint(absPosZ, absPosY, absPosX);
							errorFilename = path + "Output/PointsError" + initialMHDFilename + ".txt";
						}
					}
				}
			}*/
		}
		
		else if (gradientBasedSubVox) {


			// This method should be applied on a filtered image
			using gaussianFilterType = itk::DiscreteGaussianImageFilter<InputImageType, InputImageType>;
			auto gaussianFilter = gaussianFilterType::New();
			gaussianFilter->SetInput(image);
			gaussianFilter->SetVariance(constants::GAUSS_VARIANCE);
			gaussianFilter->Modified();
			gaussianFilter->Update();

			// the filter can be a bilateral filter
			/*using bilateralFilterType = itk::BilateralImageFilter<InputImageType, InputImageType>;
			auto bilateralFilter = bilateralFilterType::New();
			bilateralFilter->SetInput(image);
			bilateralFilter->SetDomainSigma(constants::GAUSS_VARIANCE);
			bilateralFilter->SetRangeSigma(constants::RANGE_VARIANCE);
			bilateralFilter->Modified();
			bilateralFilter->Update();*/

			// 1. Compute gradient in all 13 directions.
			// 2. Find direction of maximum gradient (equivalent to normal to the surface).
			// 3. Imporve direction to an epsilon if needed.
			// 4. Interpolate the gradient in the direction of the normal to the surface.
			CannyPixelType dx;
			CannyPixelType fPrevious;
			CannyPixelType fNext;
			indexType index;
			long deriv_window_size = 1; // How many voxels to use to compute normal vector's direction
			long interp_window_size = 3; // Amount of neighbouring voxels used for interpolating gradient
			float norm;
			float maxDx;
			long normalX, normalY, normalZ;
			indexType normalDir;

			//InputImageType::Pointer filteredImage = gaussianFilter->GetOutput(); 
			//InputImageType::Pointer filteredImage = bilateralFilter->GetOutput();
			//InputImageType::Pointer filteredImage = itk::ReadImage<InputImageType>("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Input/" + initialMHDFilename + ".mhd");
			InputImageType::Pointer filteredImage = image;

			double offsetX, offsetY, offsetZ;

			// ITK index def : "ITK assumes the first element of an index is the fastest moving index" so i tried k -> j -> i

			for (long k = 0; k < size[2]; k++) {
				for (long j = 0; j < size[1]; j++) {
					for (long i = 0; i < size[0]; i++) {
						index[0] = i;
						index[1] = j;
						index[2] = k;

						if (cannyImage->GetPixel(index) != 0.0f) {

							maxDx = 0;
							normalDir[0] = 0; normalDir[1] = 0; normalDir[2] = 0;

							float meanX = 0.0f;
							float meanY = 0.0f;
							float meanZ = 0.0f;
							for (long z = -1; z <= 1; z++) {
								for (long y = -1; y <= 1; y++) {
									for (long x = 0; x <= 1; x++) {

										// 1. Iterate over half of the neighboring cube (13 directions)
										if ((x > 0) || (x == 0 && y > 0) || (x == 0 && y == 0 && z > 0)) {

											float dx = centralDifference(filteredImage, index, {{x,y,z}}, region); // Why filteredImage here ?

											// Issue : sometimes (0, 0, 0) is the max dir
											if (abs(dx) > abs(maxDx)) {
												maxDx = dx;
												normalDir[0] = x * copysign(1.0, dx);
												normalDir[1] = y * copysign(1.0, dx);
												normalDir[2] = z * copysign(1.0, dx);
											}
										}
									}
								}
							}
							// 2. Compute direction of normal vector
							// 3. Refine direction
							// 4. Interpolate gradient in the direction of the normal to the surface

							// 4a. First we test interpolating in the direction of maximum gradient (without refinement of the normal vector)
							double maxGradient = maxDx;
							double maxGradientPos = 0;
							std::vector<double> X, Y;
							X.reserve(2 * interp_window_size + 1);
							Y.reserve(2 * interp_window_size + 1);

							for (int l = -interp_window_size; l <= interp_window_size; l++) {
								// Pas oublier de sample à intervale régulier quand on utilisera la vraie direction de la normale
								norm = vectorNorm(normalDir);
								double position = l * constants::VOXEL_SIZE * norm;
								X.push_back(position);
								// Pas normal que ici le calcul de gradient ne prenne jamais en compte les valeurs intermédiaires, probably means that a float was cast into an int and .75 became 0

								double gradient = centralDifference(filteredImage,
									{ {index[0] + l * normalDir[0], index[1] + l * normalDir[1], index[2] + l * normalDir[2]} },
									normalDir,
									region);
								Y.push_back(gradient);

								if (abs(gradient) > maxGradient) {
									maxGradient = abs(gradient);
									maxGradientPos = position;
								}
							}
							tk::spline s(X,Y);
							float maxGradientOffset = findMaxGradientOffset(s,norm, maxGradientPos);
							//std::cout << "X : " << maxGradientOffset << std::endl;

							//convertSphToCartesian(&offsetX, &offsetY, &offsetZ, norm, normalDir, maxGradientOffset); // Useless
							offsetToCartesian(&offsetX, &offsetY, &offsetZ, maxGradientOffset, normalDir);

							float absXPos = index[0] * image->GetSpacing()[0] + image->GetOrigin()[0] + offsetX;
							float absYPos = -((index[1]) * image->GetSpacing()[1] + image->GetOrigin()[1] + offsetY); // Pas oublier le - !!!
							float absZPos = index[2] * image->GetSpacing()[2] + image->GetOrigin()[2] + offsetZ;
							points->InsertNextPoint(absZPos, absYPos, absXPos);

							//std::cout << " OffsetX = " << offsetX << " OffsetY = " << offsetY << " OffsetZ = " << offsetZ << std::endl;
							//if (abs(offsetX) > 0.1 * deriv_window_size || abs(offsetY) > 0.1 * deriv_window_size || abs(offsetZ) > 0.1 * deriv_window_size) {
							if (index[0] == 53 && index[1] == 28) {
								std::cout << "Offsets : (" << offsetX << ", " << offsetY << ", " << offsetZ
									<< ") at position (" << absXPos - offsetX << ", " << absYPos + offsetY << ", " << absZPos - offsetZ << ")" 
									<< " at index " << index << std::endl;
								std::cout << "max gradient = " << maxGradient << " maxGradientOffset : " << maxGradientOffset << " in dir " << normalDir << std::endl;
								printSpline(s, norm, maxGradientPos, interp_window_size);
								std::cout << std::endl;
							}
						}
						//std::cout << "x : " << index[0] << " y : " << index[1] << " z : " << index[2] << std::endl;
					}
				}
			}
		}



		std::ofstream pointCloud;
		pointCloud.open(path + "Output/PointCloud" + initialMHDFilename + ".txt");
		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);
		polyDataBeforeMeshing->DeepCopy(polyData);


		// Test ply after subvoxel
		vtkNew<vtkPLYWriter> plyWriter;
		plyWriter->SetFileName((path + "Output/PointCloudAfterSubvoxel.ply").c_str());
		plyWriter->SetInputData(polyData);
		plyWriter->Write();


		std::cout << "End of center of gravity" << std::endl;



		vtkNew<vtkPolyData> centerPolyData; // Obtained from canny, without subvoxel refinment
		centerPolyData->DeepCopy(polyData);

		// End test ply

		double currPointCoord[3];
		itk::CovariantVector< float, 3 > currDir;
		itk::CovariantVector< float, 3 > step;
		itk::CovariantVector< float, 3 > brokenDir; // BrokenDir
		brokenDir.Fill(0);

		if (false) {
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
			/*case Poisson: {
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
			}*/
			/*case PowerCrust: { // Marche pas sans subvox refinement
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
			}*/
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

			case Delaunay: {
				vtkNew<vtkDelaunay2D> delaunay2D;
				delaunay2D->SetInputData(polyData);
				delaunay2D->Update();
				stlFilename = path + "Output/Delaunay/" + initialMHDFilename + "Delaunay3D.stl";
				compareFileName = path + "Output/Delaunay/" + initialMHDFilename + "CompDelaunay3D.vtk";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(delaunay2D->GetOutputPort());
				stlWriter->Write();

				break;
			}
		}
	}


	/*if (gradientBasedSubVox) {

		// This method should be applied on a filtered image
		using gaussianFilterType = itk::DiscreteGaussianImageFilter<InputImageType, InputImageType>;
		auto gaussianFilter = gaussianFilterType::New();
		gaussianFilter->SetInput(image);
		gaussianFilter->SetVariance(constants::GAUSS_VARIANCE);
		gaussianFilter->Modified();
		gaussianFilter->Update();

		// 1. Compute gradient in all 13 directions.
		// 2. Find direction of maximum gradient (equivalent to normal to the surface).
		// 3. Imporve direction to an epsilon if needed.
		// 4. Interpolate the gradient in the direction of the normal to the surface.
		CannyPixelType dx;
		CannyPixelType fPrevious;
		CannyPixelType fNext;
		indexType index;
		long deriv_window_size = 1; // How many voxels to use to compute normal vector's direction
		long interp_window_size = 5; // Amount of neighbouring voxels used for interpolating gradient
		float norm;
		float maxDx;
		long normalX, normalY, normalZ;
		indexType normalDir;
		InputImageType::Pointer filteredImage = gaussianFilter->GetOutput();
		//InputImageType::Pointer filteredImage = itk::ReadImage<InputImageType>("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Input/" + initialMHDFilename + ".mhd");

		double offsetX, offsetY, offsetZ;

		vtkNew<vtkPoints> newPoints;

		// We iterate over every point of the polydata
		std::array<double, 3> pointPosition;
		double p[3];

		std::cout << "Start post-mesh refinement" << std::endl;
		for (int i = 0; i < targetPolyData->GetNumberOfPoints(); i++) {
			targetPolyData->GetPoint(i, p);
			pointPosition = polyDataCoordToImageCoord({ p[0],p[1],p[2] }, image->GetSpacing(), image->GetOrigin());
			index[0] = pointPosition[0];
			index[1] = pointPosition[1];
			index[2] = pointPosition[2];

			maxDx = 0;
			normalDir[0] = 0; normalDir[1] = 0; normalDir[2] = 0;

			float meanX = 0.0f;
			float meanY = 0.0f;
			float meanZ = 0.0f;
			for (long z = -1; z <= 1; z++) {
				for (long y = -1; y <= 1; y++) {
					for (long x = 0; x <= 1; x++) {

						// 1. Iterate over half of the neighboring cube (13 directions)
						// Mettre le calcul du gradient en fonction
						if ((x > 0) || (x == 0 && y > 0) || (x == 0 && y == 0 && z > 0)) {

							float dx = centralDifference(filteredImage, index, { {x,y,z} }, region);

							// Issue : (0, 0, 0) is the max dir
							if (abs(dx) > abs(maxDx)) {
								maxDx = dx;
								normalDir[0] = x * copysign(1.0, dx);
								normalDir[1] = y * copysign(1.0, dx);
								normalDir[2] = z * copysign(1.0, dx);
							}
						}
					}
				}
			}
			// 2. Compute direction of normal vector
			// 3. Refine direction
			// 4. Interpolate gradient in the direction of the normal to the surface

			// 4a. First we test interpolating in the direction of maximum gradient (without refinement of the normal vector)
			double maxGradient = maxDx;
			double maxGradientPos = 0;
			std::vector<double> X, Y;
			X.reserve(2 * interp_window_size + 1);
			Y.reserve(2 * interp_window_size + 1);

			for (int l = -interp_window_size; l <= interp_window_size; l++) {
				// Pas oublier de sample à intervale régulier quand on utilisera la vraie direction de la normale
				norm = vectorNorm(normalDir);
				double position = l * constants::VOXEL_SIZE * norm;
				X.push_back(position);
				double gradient = centralDifference(filteredImage,
					{ {index[0] + l * normalDir[0], index[1] + l * normalDir[1], index[2] + l * normalDir[2]} },
					normalDir,
					region);
				Y.push_back(gradient);
				if (abs(gradient) > maxGradient) {
					maxGradient = abs(gradient);
					maxGradientPos = position;
				}
			}
			tk::spline s(X, Y);
			float maxGradientOffset = findMaxGradientOffset(s, norm, maxGradientPos);
			//std::cout << "X : " << maxGradientOffset << std::endl;

			//convertSphToCartesian(&offsetX, &offsetY, &offsetZ, norm, normalDir, maxGradientOffset); // Useless
			offsetToCartesian(&offsetX, &offsetY, &offsetZ, maxGradientOffset, normalDir);

			float absXPos = index[0] * image->GetSpacing()[0] + image->GetOrigin()[0] + offsetX;
			float absYPos = -((index[1]) * image->GetSpacing()[1] + image->GetOrigin()[1] + offsetY); // Pas oublier le - !!!
			float absZPos = index[2] * image->GetSpacing()[2] + image->GetOrigin()[2] + offsetZ;
			newPoints->InsertNextPoint(absZPos, absYPos, absXPos);

			//std::cout << " OffsetX = " << offsetX << " OffsetY = " << offsetY << " OffsetZ = " << offsetZ << std::endl;
			//if (abs(offsetX) > 0.1 * deriv_window_size || abs(offsetY) > 0.1 * deriv_window_size || abs(offsetZ) > 0.1 * deriv_window_size) {
			if (index[0] == 2 && index[1] == 348 && index[2] == 101) {
				std::cout << "Offsets : (" << offsetX << ", " << offsetY << ", " << offsetZ
					<< ") at position (" << absXPos - offsetX << ", " << absYPos + offsetY << ", " << absZPos - offsetZ << ")"
					<< " at index " << index << std::endl;
				std::cout << "max gradient = " << maxGradient << " maxGradientOffset : " << maxGradientOffset << " in dir " << normalDir << std::endl;
				printSpline(s, norm, maxGradientPos, interp_window_size);
				std::cout << std::endl;
			}
			//std::cout << "x : " << index[0] << " y : " << index[1] << " z : " << index[2] << std::endl;
		}
		targetPolyData->SetPoints(newPoints);
		std::cout << "End post mesh refinement" << std::endl;
	}*/
	//Toute cette partie semble de pas fonctionner mais on s'y attendait c'était juste un test



	//Load reference STL
	vtkNew<vtkSTLReader> referenceReader;
	//std::string referenceSTL = path + "Input/InitialModel.stl";
	std::string referenceSTL = path + "Input/" + referenceFilename + ".stl";
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
	//PolyDataToImageData(targetPolyData);        Je sais plus pourquoi on faisait ça mais ça crash quand je fais le centre de gravité (comprends pas pourquoi)
	


	// End rotate reference

	vtkNew<vtkCleanPolyData> referencePolyData;
	//referencePolyData->SetInputData(referenceReader->GetOutput());
	referencePolyData->SetInputData(transformToRef->GetOutput());
	referencePolyData->Update();
	std::cout << "Reference Loaded" << std::endl;












	// Test implicit distance as replacement for vtkDistancePolydataFilter

	vtkNew<vtkTransform> transCylinder; // Au lieu de faire ceci, modifier le stl serait bcp mieux
	transCylinder->RotateX(90);
	vtkNew<vtkTransformPolyDataFilter> transformCylinderToRef;
	transformCylinderToRef->SetTransform(transCylinder);
	transformCylinderToRef->SetInputData(referencePolyData->GetOutput());
	transformCylinderToRef->Update();

	vtkNew<vtkTransform> transRotatedCube;
	transRotatedCube->RotateY(90);
	vtkNew<vtkTransformPolyDataFilter> transformRotatedCubeToRef;
	transformRotatedCubeToRef->SetTransform(transRotatedCube);
	transformRotatedCubeToRef->SetInputData(referencePolyData->GetOutput());
	transformRotatedCubeToRef->Update();


	vtkNew<vtkImplicitPolyDataDistance> implicitPolyDataDistance;
	implicitPolyDataDistance->SetInput(transformCylinderToRef->GetOutput()); // If need for rotation, like the cylinder for example
	//implicitPolyDataDistance->SetInput(transformRotatedCubeToRef->GetOutput()); // If need for rotation, like the rotated cube for example
	//implicitPolyDataDistance->SetInput(referencePolyData->GetOutput()); // If no need for rotation

	vtkNew<vtkFloatArray> distances;
	distances->SetNumberOfComponents(1);
	distances->SetName("Distances");

	for (vtkIdType pointId = 0; pointId < polyDataBeforeMeshing->GetNumberOfPoints(); pointId++) {
		double p[3];
		polyDataBeforeMeshing->GetPoints()->GetPoint(pointId, p);
		float signedDistance = implicitPolyDataDistance->EvaluateFunction(p);
		distances->InsertNextValue(signedDistance);
	}
	polyDataBeforeMeshing->GetPointData()->SetScalars(distances);

	vtkNew<vtkPolyDataWriter> vtkTestDistanceWriter;

	vtkTestDistanceWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Output/testImplicitDistance.vtk");
	vtkTestDistanceWriter->SetInputData(polyDataBeforeMeshing);
	vtkTestDistanceWriter->Write();


	// End test implicit distance






	vtkNew<vtkCleanPolyData> targetCleanPolyData;
	targetCleanPolyData->SetInputData(targetPolyData);

	vtkNew<vtkVertexGlyphFilter> glyphFilter;
	glyphFilter->SetInputData(polyDataBeforeMeshing);
	glyphFilter->Update();
	
	vtkNew<vtkPolyVertex> polyVertex;
	polyVertex->GetPointIds()->SetNumberOfIds(polyDataBeforeMeshing->GetNumberOfPoints());
	for (auto i = 0; i < polyDataBeforeMeshing->GetNumberOfPoints(); i++) {
		polyVertex->GetPointIds()->SetId(i, i);
	}

	auto vertexes = vtkSmartPointer<vtkCellArray>::New();
	vertexes->InsertNextCell(polyVertex);
	polyDataBeforeMeshing->SetPolys(vertexes);

	/*auto vertexes = vtkSmartPointer<vtkCellArray>::New();
	for (int i = 0; i < polyDataBeforeMeshing->GetNumberOfPoints(); i++) {
		vtkIdType pid[1];
		pid[0] = i;
		vertexes->InsertNextCell(1, pid);
	}
	polyDataBeforeMeshing->SetVerts(vertexes);*/
	


	vtkNew<vtkCleanPolyData> targetCleanPolyDataBeforeMeshing;
	targetCleanPolyDataBeforeMeshing->SetInputData(polyDataBeforeMeshing);
	targetCleanPolyDataBeforeMeshing->Update();


	vtkNew<vtkPolyDataWriter> vtkWriter;

	// Test to remove instantly -> It shows that the filter output is not empty
	vtkWriter->SetFileName("C:/Users/xris/Desktop/RobinStuff/Surface/build/Release/Output/glyphFilterOutput.vtk");
	vtkWriter->SetInputConnection(targetCleanPolyDataBeforeMeshing->GetOutputPort());
	vtkWriter->Write();


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
	//targetCleanPolyData->SetInputData(icpTransformFilter->GetOutput());   // If want to use icp
	targetCleanPolyData->SetInputData(alignTestPolyData);					// If no icp



	// Compute distance between vertices of the reconstructed surface and the reference mesh
	std::cout << "Compute distance with reference : Start" << std::endl;

	vtkNew<vtkDistancePolyDataFilter> distanceFilter;
	distanceFilter->SetInputConnection(1, referencePolyData->GetOutputPort());
	distanceFilter->SetInputConnection(0, targetCleanPolyData->GetOutputPort());        // If want to compare points error after meshing
	//distanceFilter->SetInputConnection(0, targetCleanPolyDataBeforeMeshing->GetOutputPort()); //  If want to compare points error before meshing
	
	//distanceFilter->SetInputData(0, targetPolyData);
	distanceFilter->SetSignedDistance(false);
	distanceFilter->Update();

	
	vtkWriter->SetFileName(compareFileName.c_str());
	vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
	vtkWriter->Write();

	std::cout << "Compute distance with reference : End" << std::endl;


	// Write results as csv file

	vtkPolyData* polyDataToCSV;
	polyDataToCSV = distanceFilter->GetOutput(); // If want the result of polyDataDistanceFilter
	//polyDataToCSV->DeepCopy(polyDataBeforeMeshing); // If want the result of the implicit distance function, PAS OUBLIER DE CHANGER LE NOM DU CSV

	vtkNew<vtkDelimitedTextWriter> vtkCSVWriter;
	vtkNew<vtkTable> table;
	vtkNew<vtkFloatArray> col;
	col->SetName("Distance");
	table->AddColumn(col);
	size_t nbPoints = polyDataToCSV->GetNumberOfPoints();
	table->SetNumberOfRows(static_cast<vtkIdType>(nbPoints));
	double distanceValue[3];
	for (vtkIdType r = 0; r < table->GetNumberOfRows(); r++) {
		polyDataToCSV->GetPointData()->GetScalars()->GetTuple(r, distanceValue);
		table->SetValue(r, 0, distanceValue[0]);
	}
	//std::string distanceFilename = "C:/Users/xris/Desktop/RobinStuff/Spreadsheets/" + geometryString + algoToString[reco] + ".csv";
	std::string distanceFilename = "C:/Users/xris/Desktop/RobinStuff/Spreadsheets/cylinder02BeforeMeshing.csv";

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
