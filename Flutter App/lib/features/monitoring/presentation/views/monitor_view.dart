import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter_mjpeg/flutter_mjpeg.dart';
import 'package:http/http.dart' as http; // ← مضافة لفحص الاتصال بالسيرفر
import 'package:permission_handler/permission_handler.dart';
import 'package:security_app/constant.dart';

class MonitorView extends StatefulWidget {
  const MonitorView({Key? key}) : super(key: key);

  @override
  State<MonitorView> createState() => _MonitorViewState();
}

class _MonitorViewState extends State<MonitorView> {
  bool isMonitoring = false;
  bool isError = false;
  bool isLoading = false;

  final String videoUrl = 'http://192.168.100.3:5000/video_feed';

  Future<bool> _checkPermissions() async {
    final cameraStatus = await Permission.camera.status;
    final storageStatus = await Permission.storage.status;

    if (!cameraStatus.isGranted || !storageStatus.isGranted) {
      final status = await [Permission.camera, Permission.storage].request();
      return status[Permission.camera]?.isGranted == true &&
          status[Permission.storage]?.isGranted == true;
    }
    return true;
  }

  Future<bool> _checkInternetConnection() async {
    final connectivityResult = await Connectivity().checkConnectivity();
    return connectivityResult != ConnectivityResult.none;
  }

  Future<bool> _checkServerAvailable() async {
    try {
      final response = await http
          .get(Uri.parse(videoUrl))
          .timeout(Duration(seconds: 60)); // ← هنا بنصبر 15 ثانية
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<void> startMonitoring() async {
    try {
      setState(() {
        isLoading = true;
        isError = false;
      });

      if (!await _checkInternetConnection()) {
        throw Exception('لا يوجد اتصال بالإنترنت');
      }

      if (!await _checkPermissions()) {
        throw Exception('تم رفض الأذونات المطلوبة');
      }

      if (!await _checkServerAvailable()) {
        throw Exception('تعذر الاتصال بالسيرفر. حاول مرة أخرى لاحقاً.');
      }

      setState(() {
        isMonitoring = true;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isError = true;
        isLoading = false;
        isMonitoring = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('خطأ: ${e.toString()}')),
      );
    }
  }

  void stopMonitoring() {
    setState(() {
      isMonitoring = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: kPrimaryColor,
        title: Text('مراقبة مباشرة'),
        centerTitle: true,
      ),
      body: Center(
        child: isLoading
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('جاري الاتصال بالسيرفر...'),
                ],
              )
            : isError
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.error_outline, color: Colors.red, size: 50),
                      SizedBox(height: 20),
                      Text('فشل في الاتصال بالسيرفر'),
                      SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: startMonitoring,
                        child: Text('إعادة المحاولة'),
                      ),
                    ],
                  )
                : isMonitoring
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          AspectRatio(
                            aspectRatio: 16 / 9,
                            child: Mjpeg(
                              stream: videoUrl,
                              isLive: true,
                              error: (context, error, stack) {
                                return Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(Icons.error,
                                          size: 40, color: Colors.red),
                                      SizedBox(height: 10),
                                      Text('فشل في تحميل البث'),
                                      Text(error.toString(),
                                          style: TextStyle(fontSize: 12)),
                                    ],
                                  ),
                                );
                              },
                            ),
                          ),
                          SizedBox(height: 20),
                          ElevatedButton(
                            onPressed: stopMonitoring,
                            child: Text('إيقاف المراقبة'),
                          ),
                        ],
                      )
                    : ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(
                              horizontal: 32, vertical: 16),
                        ),
                        onPressed: startMonitoring,
                        child: Text(
                          'بدء المراقبة',
                          style: TextStyle(
                            fontSize: 18,
                            color: kPrimaryColor,
                          ),
                        ),
                      ),
      ),
    );
  }
}
