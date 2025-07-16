import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:security_app/constant.dart';
import 'package:security_app/core/utils/app_router.dart';

class Homeview extends StatefulWidget {
  const Homeview({super.key});

  @override
  State<Homeview> createState() => _HomeviewState();
}

class _HomeviewState extends State<Homeview> {
  List<String> notifications = [];

  @override
  void initState() {
    super.initState();
    // هنا توصل بالـ SocketIO
    // مثال:
    // socket.on('violence_detected', (data) {
    //   setState(() {
    //     notifications.add(data['message']);
    //   });
    // });
    // socket.on('processing_complete', (data) {
    //   setState(() {
    //     notifications.add('Video processing complete');
    //   });
    // });
  }

  void showNotificationDialog() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Notifications'),
        content: SizedBox(
          width: double.maxFinite,
          child: ListView.separated(
            shrinkWrap: true,
            itemCount: notifications.length,
            itemBuilder: (context, index) =>
                Text("• ${notifications[index]}"),
            separatorBuilder: (context, index) =>
                const Divider(height: 10),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Close"),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kPrimaryColor,
      appBar: AppBar(
        backgroundColor: kPrimaryColor,
        elevation: 0,
        centerTitle: true,
        title: const Text(
          "Vioguard",
          style: TextStyle(
            color: Colors.white,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications, color: Colors.white),
            onPressed: showNotificationDialog,
          ),
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CustomButton(
                text: "Start Monitoring",
                icon: Icons.videocam,
                onPressed: () {
                  GoRouter.of(context).push(AppRouter.kMonitorView);
                },
              ),
              const SizedBox(height: 24),
              CustomButton(
                text: "Analyze Video",
                icon: Icons.upload_file,
                onPressed: () {
                  GoRouter.of(context).push(AppRouter.kAnalyzeVideoView);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}


class CustomButton extends StatelessWidget {
  final String text;
  final IconData icon;
  final VoidCallback onPressed;

  const CustomButton({
    super.key,
    required this.text,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        icon: Icon(icon, color: kPrimaryColor, size: 24),
        label: Text(
          text,
          style: const TextStyle(
            fontSize: 18,
            color: kPrimaryColor,
            fontWeight: FontWeight.bold,
          ),
        ),
        onPressed: onPressed,
      ),
    );
  }
}