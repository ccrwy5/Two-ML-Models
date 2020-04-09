//
//  ImageViewController.swift
//  ML-Vision
//
//  Created by DALE MUSSER on 12/12/17.
//  Updated 10/26/18 for Xcode 10.0
//  Copyright © 2017 Tech Innovator. All rights reserved.
//
// http://www.wolfib.com/Image-Recognition-Intro-Part-1/
// https://developer.apple.com/machine-learning/

import UIKit
import CoreML
import Vision

class ImageViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var leftTextView: UITextView!
    @IBOutlet weak var rightTextView: UITextView!
    
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        leftTextView.text = ""
        rightTextView.text = ""
        activityIndicator.hidesWhenStopped = true
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func cameraSelected(_ sender: Any) {
        takePhotoWithCamera()
    }
    

    @IBAction func photoLibrarySelected(_ sender: Any) {
        pickPhotoFromLibrary()
    }
    
    func takePhotoWithCamera() {
        if (!UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera)) {
            let alertController = UIAlertController(title: "No Camera", message: "The device has no camera.", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "OK", style: .default, handler: nil)
            alertController.addAction(okAction)
            present(alertController, animated: true, completion: nil)
        } else {
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .camera
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func pickPhotoFromLibrary() {
        imagePicker.allowsEditing = false
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            leftTextView.text = ""
            rightTextView.text = ""
            
            guard let ciImage = CIImage(image: pickedImage) else {
                displayStringForLeft(string: "Unable to convert image to CIImage.");
                return
            }
            
            detectScene(image: ciImage)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func displayStringForLeft(string: String) {
        leftTextView.text = leftTextView.text + string + "\n";
    }
    
    func displayStringForRight(string: String){
        rightTextView.text = rightTextView.text + string + "\n";
    }
    
    func detectScene(image: CIImage) {
        displayStringForLeft(string: "detecting scene...")
        displayStringForRight(string: "detecting scene...")

                
         guard let secondModel = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
         displayStringForRight(string: "Can't load ML model.")
         return
         }
        
        let request2 = VNCoreMLRequest(model: secondModel) { [weak self] request2, error in
            guard let secondResults = request2.results as? [VNClassificationObservation],
                let _ = secondResults.first else {
                    self?.displayStringForRight(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                for result in secondResults {
                    self?.displayStringForRight(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                }
            }
        }
        
        activityIndicator.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler2 = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler2.perform([request2])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayStringForRight(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
        
/***********************************************************************************************/
        guard let model = try? VNCoreMLModel(for: VGG16().model) else {
            displayStringForLeft(string: "Can't load ML model.")
            return
        }
        
        
        // Create a Vision request with completion handler
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayStringForLeft(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                for result in results {
                    self?.displayStringForLeft(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                }
            }
        }
        
        activityIndicator.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayStringForLeft(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
    }

}



