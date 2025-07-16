import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;

class MonitoringService {
  final String videoUrl = 'http://192.168.1.5:5000/video_feed';

  /// يتحقق من الاتصال بالإنترنت
  Future<bool> checkInternetConnection() async {
    final connectivityResult = await Connectivity().checkConnectivity();
    return connectivityResult != ConnectivityResult.none;
  }

  /// يتحقق من أذونات الكاميرا والتخزين
  Future<bool> checkPermissions() async {
    final cameraStatus = await Permission.camera.status;
    final storageStatus = await Permission.storage.status;

    if (!cameraStatus.isGranted || !storageStatus.isGranted) {
      final status = await [Permission.camera, Permission.storage].request();
      return status[Permission.camera]?.isGranted == true &&
          status[Permission.storage]?.isGranted == true;
    }
    return true;
  }

  /// يتحقق من أن الفيديو متاح من السيرفر
  Future<bool> isVideoFeedAvailable() async {
    try {
      final response = await http.get(Uri.parse(videoUrl));
      return response.statusCode == 200;
    } catch (e) {
      print('Video feed check failed: $e');
      return false;
    }
  }
}
