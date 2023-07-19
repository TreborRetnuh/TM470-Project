package com.example.tm470fishidentifier;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.example.tm470fishidentifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    // Set variables
    Button gallery; // Don’t need camera at the moment
    ImageView imageView;
    TextView result;
    int imageSize = 256;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        gallery = findViewById(R.id.gallery);

        //result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Assign gallery button function
        gallery.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                galleryActivityResultLauncher.launch(intent); // no request code?
            }

    // Image classifier function
    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Create inputs.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            // Add r,g and b values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Run model and get result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // Find the biggest confidence index.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Bream", "Sturgeon", "Eel", "Barbel", "Silver Bream", "Crucian Carp",
                    "Grass Carp", "Common carp", "Pike", "Gudgeon", "Ruffe", "Chub", "Ide",
                    "Perch", "Roach", "Brown Trout", "Zander", "Rudd", "Catfish", "Tench"};
            result.setText(classes[maxPos]);


            // Close model
            model.close();
        } catch (IOException e) {
        // TODO Handle the exception
        }

    }

            // Launch gallery activity
            final ActivityResultLauncher<Intent> galleryActivityResultLauncher = registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(),
                    new ActivityResultCallback<ActivityResult>() {
                        @Override
                        public void onActivityResult(ActivityResult result) {
                            if (result.getResultCode() == Activity.RESULT_OK) {
                                // no request codes?
                                //Uri image_uri = result.getData().getData();
                                //imageView.setImageURI(image_uri);

                                // convert to bitmap
                                assert result.getData() != null;
                                Uri dat = result.getData().getData(); //image.uri:
                                Bitmap image = null;
                                try {
                                    image = MediaStore.Images.Media.getBitmap(getContentResolver(), dat);
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                                imageView.setImageBitmap(image);

                                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                                classifyImage(image);

                            }
                        }
                    });
        });


    }
}