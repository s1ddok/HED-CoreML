//
//  UIImage+Resize.swift
//  HED-CoreML
//
//  Created by Andrey Volodin on 03.07.17.
//  Copyright © 2017 s1ddok. All rights reserved.
//

import UIKit

public extension UIImage {
    public func resized(width: Int, height: Int) -> UIImage {
        guard width > 0 && height > 0 else {
            fatalError("Dimensions must be over 0.")
        }
        
        let newSize = CGSize(width: width, height: height)
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0);
        self.draw(in: CGRect(origin: CGPoint.zero, size: newSize))
        let newImage: UIImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return newImage
    }
}
