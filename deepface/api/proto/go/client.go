package main

import (
	"context"
	"flag"
	"log"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/dal-papa/deepface/proto"
)

var flagImg = flag.String("img", "", "Path to the image file")

func main() {
	ctx := context.Background()
	flag.Parse()
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewDeepFaceServiceClient(conn)
	res, err := client.Analyze(ctx, &pb.AnalyzeRequest{
		ImageUrl: *flagImg,
		Actions: []pb.AnalyzeRequest_Action{
			pb.AnalyzeRequest_AGE,
			pb.AnalyzeRequest_GENDER,
			pb.AnalyzeRequest_RACE,
		},
	})
	if err != nil {
		log.Fatalf("error calling Analyze: %v", err)
	}
	log.Printf("Analyze response: %+v", res)
}
