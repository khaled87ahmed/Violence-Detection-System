import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:security_app/constant.dart';
import 'package:security_app/core/utils/app_router.dart';
import 'package:security_app/core/utils/assets_data.dart';

class SplashViewBody extends StatefulWidget {
  const SplashViewBody({super.key});

  @override
  State<SplashViewBody> createState() => _SplashViewBodyState();
}

class _SplashViewBodyState extends State<SplashViewBody> {
  @override
  void initState() {
    
    super.initState();
    navigateToHome();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // الخلفية (ممكن صورة أو لون)
        Container(
          decoration: BoxDecoration(
            color: kPrimaryColor,
            image: DecorationImage(
              image: AssetImage(AssetsData.splashpic),
              fit: BoxFit.cover,
            ),
          ),
        ),
        // تأثير الضبابية
        BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 5.0, sigmaY: 5.0),
          child: Container(
              color: kPrimaryColor.withOpacity(0.4) // إضافة طبقة شفافة
              ),
        ),
        // اللوجو والنص
        Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.security,
                size: 100,
                color: Colors.white,
              ),
              SizedBox(height: 20),
              Text(
                'Vioguard',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void navigateToHome() {
    Future.delayed(Duration(seconds: 3), () {
      GoRouter.of(context).push(AppRouter.kloginView);
    });
  }
}
