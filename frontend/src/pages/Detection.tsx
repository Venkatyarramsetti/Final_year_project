import React, { useState, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, Camera, Loader2, AlertTriangle, CheckCircle } from 'lucide-react';
import Header from "@/components/Header";

interface Detection {
  class_name: string;
  confidence: number;
  category: string;
  bounding_box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface DetectionResult {
  total_detections: number;
  hazardous_count: number;
  healthy_count: number;
  overall_assessment: string;
  detections: Detection[];
}

const Detection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        setError(null);
        setResults(null);
        
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
          setImagePreview(e.target?.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        setError('Please select a valid image file.');
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleDetection = async () => {
    if (!selectedImage) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setResults(data.results);
      } else {
        throw new Error('Detection failed');
      }
    } catch (err) {
      setError(`Error: ${err instanceof Error ? err.message : 'Unknown error occurred'}`);
      console.error('Detection error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const resetDetection = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Hazard Detection
            </h1>
            <p className="text-lg text-gray-600">
              Upload an image to detect and classify potential hazards using AI
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Upload Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Image
                </CardTitle>
                <CardDescription>
                  Select an image to analyze for potential hazards
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                  {imagePreview ? (
                    <div className="space-y-4">
                      <img
                        src={imagePreview}
                        alt="Selected"
                        className="max-w-full h-48 object-contain mx-auto rounded"
                      />
                      <p className="text-sm text-gray-600">{selectedImage?.name}</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Camera className="h-12 w-12 text-gray-400 mx-auto" />
                      <p className="text-gray-600">No image selected</p>
                    </div>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button onClick={handleUploadClick} variant="outline" className="flex-1">
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Image
                  </Button>
                  {selectedImage && (
                    <Button onClick={resetDetection} variant="outline">
                      Clear
                    </Button>
                  )}
                </div>

                <Button 
                  onClick={handleDetection} 
                  disabled={!selectedImage || isLoading}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Camera className="h-4 w-4 mr-2" />
                      Start Detection
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Results Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Detection Results
                </CardTitle>
                <CardDescription>
                  Analysis results and hazard classification
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <Alert variant="destructive" className="mb-4">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {results ? (
                  <div className="space-y-4">
                    {/* Overall Assessment */}
                    <Alert variant={results.overall_assessment === 'Hazardous' ? 'destructive' : 'default'}>
                      {results.overall_assessment === 'Hazardous' ? (
                        <AlertTriangle className="h-4 w-4" />
                      ) : (
                        <CheckCircle className="h-4 w-4" />
                      )}
                      <AlertDescription>
                        <strong>Overall Assessment: {results.overall_assessment}</strong>
                      </AlertDescription>
                    </Alert>

                    {/* Summary Stats */}
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="bg-blue-50 p-3 rounded">
                        <div className="text-2xl font-bold text-blue-600">
                          {results.total_detections}
                        </div>
                        <div className="text-sm text-blue-600">Total Objects</div>
                      </div>
                      <div className="bg-red-50 p-3 rounded">
                        <div className="text-2xl font-bold text-red-600">
                          {results.hazardous_count}
                        </div>
                        <div className="text-sm text-red-600">Hazardous</div>
                      </div>
                      <div className="bg-green-50 p-3 rounded">
                        <div className="text-2xl font-bold text-green-600">
                          {results.healthy_count}
                        </div>
                        <div className="text-sm text-green-600">Safe</div>
                      </div>
                    </div>

                    {/* Detailed Detections */}
                    {results.detections.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="font-semibold">Detected Objects:</h4>
                        <div className="max-h-40 overflow-y-auto space-y-2">
                          {results.detections.map((detection, index) => (
                            <div
                              key={index}
                              className={`p-2 rounded border ${
                                detection.category === 'Hazardous'
                                  ? 'border-red-200 bg-red-50'
                                  : 'border-green-200 bg-green-50'
                              }`}
                            >
                              <div className="flex justify-between items-center">
                                <span className="font-medium">{detection.class_name}</span>
                                <span className={`text-xs px-2 py-1 rounded ${
                                  detection.category === 'Hazardous'
                                    ? 'bg-red-200 text-red-800'
                                    : 'bg-green-200 text-green-800'
                                }`}>
                                  {detection.category}
                                </span>
                              </div>
                              <div className="text-sm text-gray-600">
                                Confidence: {(detection.confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <AlertTriangle className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                    <p>Upload and analyze an image to see results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Detection;
