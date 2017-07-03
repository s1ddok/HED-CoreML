//
//  ViewController.swift
//  HED-CoreML
//
//  Created by Andrey Volodin on 02.07.17.
//  Copyright © 2017 s1ddok. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {
    
    enum SelectedModel: Int {
        case fuse = 0, dsn5, dsn4, dsn3, dsn2, dsn1
        
        var outputLayerName: String {
            switch self {
            case .fuse:
                return "upscore-fuse"
            case .dsn5:
                return "upscore-dsn5"
            case .dsn4:
                return "upscore-dsn4"
            case .dsn3:
                return "upscore-dsn3"
            case .dsn2:
                return "upscore-dsn2"
            case .dsn1:
                return "upscore-dsn1"
            }
        }
    }
    
    let hedMain = HED_fuse()
    let hedSO = HED_so()
    
    var selectedModel: SelectedModel = .fuse
    
    var cachedCalculationResults: [SelectedModel : UIImage] = [:]
    
    @IBOutlet weak var resultsSegmentedControl: UISegmentedControl!
    @IBOutlet weak var imageView: UIImageView!
    var inputImage: UIImage!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.inputImage = UIImage(named: "pic1.jpg")
        self.imageView.image = inputImage
    }

    @IBAction func selectedResultsChanged(_ sender: UISegmentedControl) {
        if sender.selectedSegmentIndex == 0 {
            self.imageView.image = self.inputImage
        } else {
            self.imageView.image = self.cachedCalculationResults[self.selectedModel]
        }
    }
    
    @IBAction func selectedModelChanged(_ sender: UISegmentedControl) {
        self.selectedModel = SelectedModel(rawValue: sender.selectedSegmentIndex)!
        
        if cachedCalculationResults[selectedModel] == nil {
            resultsSegmentedControl.selectedSegmentIndex = 0
            resultsSegmentedControl.setEnabled(false, forSegmentAt: 1)
            
            self.imageView.image = self.inputImage
        } else {
            resultsSegmentedControl.setEnabled(true, forSegmentAt: 1)
            if resultsSegmentedControl.selectedSegmentIndex == 1 {
                self.imageView.image = cachedCalculationResults[self.selectedModel]
            }
        }
    }
    
    @IBAction func doInferencePressed(_ sender: UIButton) {
        guard cachedCalculationResults[selectedModel] == nil else {
            return
        }
        
        // Remember the time when we started
        let startDate = Date()
        
        // Convert our image to proper input format
        // In this case we need to feed pixel buffer which is 500x500 sized.
        let inputW = 500
        let inputH = 500
        guard let inputPixelBuffer = inputImage.resized(width: inputW, height: inputH)
                                               .pixelBuffer(width: inputW, height: inputH) else {
            fatalError("Couldn't create pixel buffer.")
        }
        
        // Use different models based on what output we need
        let featureProvider: MLFeatureProvider
        switch selectedModel {
        case .fuse:
            featureProvider = try! hedMain.prediction(data: inputPixelBuffer)
        case .dsn1, .dsn2, .dsn3, .dsn4, .dsn5:
            featureProvider = try! hedSO.prediction(data: inputPixelBuffer)
        }
        
        // Retrieve results
        guard let outputFeatures = featureProvider.featureValue(for: selectedModel.outputLayerName)?.multiArrayValue else {
            fatalError("Couldn't retrieve features")
        }
        
        // Calculate total buffer size by multiplying shape tensor's dimensions
        let bufferSize = outputFeatures.shape.lazy.map { $0.intValue }.reduce(1, { $0 * $1 })
        
        // Get data pointer to the buffer
        let dataPointer = UnsafeMutableBufferPointer(start: outputFeatures.dataPointer.assumingMemoryBound(to: Double.self),
                                                     count: bufferSize)
        
        // Prepare buffer for single-channel image result
        var imgData = [UInt8](repeating: 0, count: bufferSize)
        
        // Normalize result features by applying sigmoid to every pixel and convert to UInt8
        for i in 0..<inputW {
            for j in 0..<inputH {
                let idx = i * inputW + j
                let value = dataPointer[idx]
                
                let sigmoid = { (input: Double) -> Double in
                    return 1 / (1 + exp(-input))
                }
                
                let result = sigmoid(value)
                imgData[idx] = UInt8(result * 255)
            }
        }
        
        // Create single chanel gray-scale image out of our freshly-created buffer
        let cfbuffer = CFDataCreate(nil, &imgData, bufferSize)!
        let dataProvider = CGDataProvider(data: cfbuffer)!
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let cgImage = CGImage(width: inputW, height: inputH, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: inputW, space: colorSpace, bitmapInfo: [], provider: dataProvider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        let resultImage = UIImage(cgImage: cgImage!)
        
        // Calculate the time of inference
        let endDate = Date()
        print("Inference is finished in \(endDate.timeIntervalSince(startDate)) for model: \(self.selectedModel.outputLayerName)")
        
        // Cache results
        self.cachedCalculationResults[self.selectedModel] = resultImage
        
        // Enable edge-mode results
        self.resultsSegmentedControl.setEnabled(true, forSegmentAt: 1)
    }
}

