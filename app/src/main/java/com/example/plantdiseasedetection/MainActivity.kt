package com.example.plantdiseasedetection

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.plantdiseasedetection.ml.Model2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {

    lateinit var bitmap: Bitmap
    lateinit var imgview:ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgview = findViewById(R.id.imageView)
        val arr: MutableList<String> = mutableListOf("Healthy","Powdery","Rust")

        var tv:TextView = findViewById(R.id.textView)

        var select: Button = findViewById(R.id.button)

        select.setOnClickListener(View.OnClickListener {
            var intent:Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)

        })

        var predict:Button = findViewById(R.id.button1)
        predict.setOnClickListener(View.OnClickListener {

            var resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 80,80,true)
            val model = Model2.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 80, 80, 3), DataType.FLOAT32)

            var tbuffer = TensorImage.fromBitmap(resized)
            var byteBuffer = tbuffer.buffer

            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            var max = getMax(outputFeature0.floatArray)

            tv.setText(arr[max])

// Releases model resources if no longer used.
            model.close()
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?){
        super.onActivityResult(requestCode,resultCode,data)

        imgview.setImageURI(data?.data)

        var uri: Uri?= data?.data

        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
    }

    fun getMax(arr:FloatArray) : Int{

        var ind = 0
        var min = 0.0f

        for(i in 0..2)
        {
            if(arr[i]>min)
            {
                min=arr[i]
                ind=i;
            }
        }
        return ind
    }
}